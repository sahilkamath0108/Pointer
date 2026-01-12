from google import genai
import os
import json
import asyncio
from utils.logger import logger
from utils.code_formatter import format_code_for_whatsapp, truncate_message
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class GeminiService:
    def __init__(self):
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            logger.error("Gemini API key not found in environment variables")
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        
        # Get GitHub token from environment
        self.default_github_token = os.environ.get('GITHUB_TOKEN')
        if self.default_github_token:
            logger.info("GitHub token loaded from environment variables")
        else:
            logger.warning("No GitHub token found in environment variables")
            
        logger.info("Gemini AI service initialized successfully")
        
        # Define system prompt for coding assistant
        self.system_prompt = """
            You are an AI Coding Assistant on WhatsApp.

            Your role is to help users build, manage, and improve apps or websites by:
            1. Writing clean, functional, and production-ready code
            2. Explaining programming and web development concepts clearly
            3. Providing step-by-step development guidance for tasks
            4. Debugging issues and suggesting optimal solutions
            5. Reviewing and improving existing code
            6. Managing GitHub repositories and their contents

            Always:
            - Format code using proper markdown (with syntax highlighting for WhatsApp).
            - Use the user’s past messages to understand context and give consistent follow-ups.
            - Prioritize clarity and completeness over brevity. Explain decisions and outputs when helpful.

            ---

            ### GitHub Integration (Enabled ✅)

            You have access to a suite of **GitHub tools** via the Model Context Protocol (MCP). Use them to directly assist users with GitHub workflows when requested. These include, but are not limited to:

            📁 **Repository Management**
            - `createRepo`: Create new repositories
            - `getRepo`: Fetch details of a repository
            - `deleteRepo`: Delete existing repositories

            📄 **File & Content Management**
            - `createFile`: Create or upload new files
            - `getFile`: Fetch contents of a specific file
            - `updateFile`: Modify or overwrite files
            - `deleteFile`: Remove files from a repository
            - `listRepoContent`: List files/folders at any path in the repo

            🔍 **Search & Discovery**
            - `searchRepos`: Search public GitHub repositories by keyword

            🛠️ **Branch & Commit Operations**
            - `createBranch`: Create a new branch from a reference
            - `getBranches`: List all branches in a repository
            - `createCommit`: Create a commit with file changes
            - `mergeBranches`: Merge one branch into another

            📦 **Version Control & Git Workflows**
            - `getDefaultBranch`: Fetch the default branch of a repository
            - `getLatestCommit`: Get the latest commit on a branch
            - `getRepoTree`: Retrieve the tree structure of a repository

            When the user asks for GitHub-related tasks (e.g., file creation, repo setup, commits), use these tools as appropriate.

            ⚠️ If the user’s request involves **multiple steps**, like initializing a full repo with multiple files:
            - Plan and **chain multiple function calls**.
            - Confirm each step completes successfully.
            - Follow up with a summary or next instructions.

            ---

            Be proactive, detail-oriented, and helpful — like a developer friend who writes, reviews, and deploys code on command.

        """
        
    def _make_content(self, role: str, text: str):
        return types.Content(role=role, parts=[types.Part(text=text)])

    def _format_chat_history(self, chat_history):
        """Convert chat history to list[types.Content] for Gemini"""
        if not chat_history:
            return []
        formatted = []
        for msg in chat_history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            formatted.append(self._make_content(role, msg["content"]))
        return formatted
    
    def _clean_schema(self, schema):
        """Recursively clean JSON schema by removing unsupported keys"""
        if isinstance(schema, dict):
            # Create a new dictionary to avoid modifying the original during iteration
            new_schema = {}
            for k, v in schema.items():
                if k not in ["additionalProperties", "$schema"]:
                    new_schema[k] = self._clean_schema(v)
            return new_schema
        elif isinstance(schema, list):
            return [self._clean_schema(item) for item in schema]
        else:
            return schema
    
    def _filter_unset_parameters(self, arguments, schema):
        """Filter out unset/None parameters from arguments based on schema requirements
        This is the key fix from Microsoft AutoGen PR #6125"""
        if not schema or not isinstance(schema, dict):
            return arguments
        
        # Get required parameters from schema
        required_params = schema.get('required', [])
        properties = schema.get('properties', {})
        
        # Filter arguments to only include:
        # 1. Required parameters (even if None/empty)
        # 2. Optional parameters that have actual values
        filtered_args = {}
        
        for key, value in arguments.items():
            if key in required_params:
                # Always include required parameters
                filtered_args[key] = value
            elif key in properties and value is not None and value != "":
                # Only include optional parameters that have actual values
                filtered_args[key] = value
        
        return filtered_args
    
    async def _get_mcp_tools(self, github_token, mcp_session):
        """Get MCP tools from GitHub server using provided session"""
        try:
            logger.info("Retrieving MCP tools from session...")
            mcp_tools = await mcp_session.list_tools()
            logger.info(f"Retrieved {len(mcp_tools.tools)} MCP tools")
            
            declarations = []
            for tool in mcp_tools.tools:
                # Store both the declaration and the original schema for parameter filtering
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._clean_schema(tool.inputSchema),
                    "original_schema": tool.inputSchema  # Keep original for parameter filtering
                }
                declarations.append(tool_info)
            
            return declarations
        except Exception as e:
            logger.error(f"Error getting MCP tools: {str(e)}")
            return []
    
    async def _execute_function_call(self, function_call, mcp_session, tool_schemas=None):
        """Execute a function call using MCP with parameter filtering fix and provided session"""
        try:
            logger.info(f"Executing function call: {function_call.name}")
            
            # Get the original arguments
            original_args = dict(function_call.args)
            logger.info(f"Original function arguments: {original_args}")
            
            # Apply parameter filtering fix from Microsoft AutoGen PR #6125
            filtered_args = original_args
            if tool_schemas:
                # Find the schema for this specific tool
                tool_schema = None
                for schema in tool_schemas:
                    if schema.get("name") == function_call.name:
                        tool_schema = schema.get("original_schema")
                        break
                
                if tool_schema:
                    filtered_args = self._filter_unset_parameters(original_args, tool_schema)
                    logger.info(f"Filtered function arguments: {filtered_args}")
            
            # Use provided session
            result = await mcp_session.call_tool(
                function_call.name, 
                arguments=filtered_args
            )
            
            logger.info(f"Function call {function_call.name} executed successfully")
            
            # Parse and return result
            if result.content and len(result.content) > 0:
                try:
                    # Try to parse as JSON first
                    result_data = json.loads(result.content[0].text)
                    return json.dumps(result_data, indent=2)
                except json.JSONDecodeError:
                    # Return as plain text if not JSON
                    return result.content[0].text
            else:
                return "Function executed successfully but returned no content."
                        
        except Exception as e:
            logger.error(f"Error executing function call: {str(e)}")
            return f"Error executing function: {str(e)}"
        
    def generate_response(self, user_message, github_token=None, chat_history=None):
        """
        Generate a response using Gemini API with MCP GitHub integration
        """
        try:
            logger.info("Generating AI response for user message")
            
            # Use provided token or fall back to default
            token_to_use = github_token or self.default_github_token
            
            # Run the async function to get response
            return asyncio.run(self._generate_response_async(
                user_message, token_to_use, chat_history, self.system_prompt
            ))
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error. Please try again."
    
    async def _generate_response_async(self, user_message, github_token, chat_history, enhanced_system_prompt):
        """Async version of response generation with MCP support"""
        try:
            # Prepare the conversation for Gemini
            conversation_messages = [
                self._make_content("user", enhanced_system_prompt),
                self._make_content("model", "I understand. I'm your AI coding assistant ready to help you build apps and websites. How can I assist you today?")
            ]
            
            # Add chat history if available
            if chat_history:
                formatted = self._format_chat_history(chat_history)
                conversation_messages.extend(formatted)
                logger.info(f"Using chat history with {len(formatted)} previous messages")
            
            # Add current user message
            conversation_messages.append(self._make_content("user", user_message))
            
            # Get MCP tools if GitHub token is available
            tools = []
            tool_schemas = None
            token_to_use = github_token or self.default_github_token
            mcp_session = None
            
            if token_to_use:
                try:
                    # Create MCP connection once for this operation cycle
                    logger.info("Initializing MCP GitHub server connection...")
                    server_params = StdioServerParameters(
                        command="npx.cmd",
                        args=["-y", "@modelcontextprotocol/server-github"],
                        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token_to_use}
                    )
                    
                    # Use async with to properly manage the connection lifecycle
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            logger.info("MCP session initialized successfully")
                            
                            # Store session reference for reuse in this scope
                            mcp_session = session
                            
                            # Get MCP tools using the session
                            logger.info("Attempting to load MCP tools...")
                            declarations = await self._get_mcp_tools(github_token, mcp_session)
                            if declarations:
                                # Store the full schemas for parameter filtering
                                tool_schemas = declarations
                                
                                # Create Gemini tools from declarations (without original_schema)
                                gemini_declarations = []
                                for decl in declarations:
                                    gemini_declarations.append({
                                        "name": decl["name"],
                                        "description": decl["description"],
                                        "parameters": decl["parameters"]
                                    })
                                
                                tools = [types.Tool(function_declarations=gemini_declarations)]
                                logger.info(f"Loaded {len(declarations)} MCP tools")
                            else:
                                logger.warning("No MCP tools loaded")
                            
                            # Note: Connection will be reused below for function calls within this async with block
                            # Generate response with Gemini
                            config = types.GenerateContentConfig(
                                temperature=0.4,
                                max_output_tokens=65535,
                            )
                            
                            # Add tools to config if available
                            if tools:
                                config.tools = tools
                                logger.info("Tools added to Gemini config")
                            
                            logger.info("Sending request to Gemini...")
                            response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=conversation_messages,
                                config=config,
                            )
                            print(response)
                            # Check for function calls in all parts
                            function_calls = []
                            if (response.candidates and 
                                len(response.candidates) > 0 and 
                                response.candidates[0].content.parts):
                                
                                for part in response.candidates[0].content.parts:
                                    if hasattr(part, 'function_call') and part.function_call:
                                        function_calls.append(part.function_call)
                                        logger.info(f"Function call detected: {part.function_call.name}")
                            
                            # Execute all function calls if any (reusing the same session)
                            if function_calls:
                                logger.info(f"Processing {len(function_calls)} function calls")
                                
                                try:
                                    # Execute all function calls using the same session
                                    function_results = []
                                    for function_call in function_calls:
                                        print(function_call)
                                        function_result = await self._execute_function_call(
                                            function_call, 
                                            mcp_session, 
                                            tool_schemas
                                        )
                                        function_results.append(f"{function_call.name}: {function_result}")
                                    
                                    # Combine all results
                                    all_results = "\n".join(function_results)
                                    
                                    # Generate a follow-up response with all function results
                                    follow_up_messages = conversation_messages + [
                                        self._make_content("model", f"I'll execute {len(function_calls)} function(s) for you."),
                                        self._make_content("user", f"Function results: {all_results}")
                                    ]
                                    
                                    try:
                                        follow_up_response = self.client.models.generate_content(
                                            model=self.model_name,
                                            contents=follow_up_messages,
                                            config=types.GenerateContentConfig(
                                                temperature=0.2,
                                                max_output_tokens=65535,
                                            ),
                                        )
                                        
                                        if follow_up_response.text:
                                            response_text = follow_up_response.text.strip()
                                        else:
                                            response_text = f"All functions executed successfully: {all_results}"
                                    except Exception as follow_up_error:
                                        logger.error(f"Error generating follow-up response: {str(follow_up_error)}")
                                        response_text = f"All functions executed successfully: {all_results}"
                                    
                                except Exception as function_error:
                                    logger.error(f"Error executing function calls: {str(function_error)}")
                                    response_text = f"I attempted to execute {len(function_calls)} function(s), but encountered an error: {str(function_error)}"
                                    
                            elif response.text:
                                response_text = response.text.strip()
                                logger.info("Regular text response generated")
                            else:
                                logger.warning("No text in Gemini response")
                                response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                            
                            # Apply code formatting for WhatsApp
                            response_text = format_code_for_whatsapp(response_text)
                            
                            # Truncate if too long for WhatsApp
                            response_text = truncate_message(response_text, max_length=1500)
                            
                            logger.info("Code formatting applied for WhatsApp display")
                            logger.info(f"AI response generated successfully ({len(response_text)} chars)")
                            return response_text
                            
                except Exception as e:
                    logger.error(f"Error in MCP operations: {str(e)}")
                    # Fall back to non-MCP response
                    tools = []
                    tool_schemas = None
            
            # Fallback: Generate response without MCP tools
            if not token_to_use:
                logger.info("No GitHub token available, skipping MCP tools")
            
            # Generate response with Gemini (without MCP tools in fallback case)
            config = types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=65535,
            )
            
            # Add tools to config if available (should be empty in fallback)
            if tools:
                config.tools = tools
                logger.info("Tools added to Gemini config")
            
            logger.info("Sending request to Gemini...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=conversation_messages,
                config=config,
            )
            print(response)
            
            # In fallback case, tools won't be available, so no function calls
            if response.text:
                response_text = response.text.strip()
                logger.info("Regular text response generated")
            else:
                logger.warning("No text in Gemini response")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            # Apply code formatting for WhatsApp
            response_text = format_code_for_whatsapp(response_text)
            
            # Truncate if too long for WhatsApp
            response_text = truncate_message(response_text, max_length=1500)
            
            logger.info("Code formatting applied for WhatsApp display")
            logger.info(f"AI response generated successfully ({len(response_text)} chars)")
            return response_text
            
        except Exception as e:
            logger.error(f"Error in async response generation: {str(e)}")
            # Return a user-friendly error message instead of raising
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a simpler request." 