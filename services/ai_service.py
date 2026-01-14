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
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
        
        # Get GitHub token from environment
        self.default_github_token = os.environ.get('GITHUB_TOKEN')
        if self.default_github_token:
            logger.info("GitHub token loaded from environment variables")
        else:
            logger.warning("No GitHub token found in environment variables")
        
        # Get Netlify token from environment
        self.netlify_token = os.environ.get('NETLIFY_API_KEY')
        if self.netlify_token:
            logger.info("Netlify token loaded from environment variables")
        else:
            logger.warning("No Netlify token found in environment variables")
            
        logger.info("Gemini AI service initialized successfully")
        
        # Define system prompt for coding assistant
        self.system_prompt = """
You are an AI Coding Assistant on WhatsApp.

Your role is to help users build, manage, and improve apps or websites.

CRITICAL - WhatsApp Response Guidelines:
- Keep responses SHORT and TO THE POINT (max 2-3 sentences)
- NO code blocks or long code snippets - users cannot read them on WhatsApp
- Use bullet points only when absolutely necessary
- Provide concise summaries instead of detailed explanations
- Ask clarifying questions if the user's intent is unclear
- Confirm actions before executing them
- Report results in 1-2 sentences

INTERNAL LOOP BEHAVIOR:
- You will internally loop and refine your understanding until you are confident about what the user wants
- Only provide a final response to the user when you are satisfied you understand their complete goal
- Use internal reasoning to clarify ambiguities, plan multi-step operations, and validate your approach
- Do NOT show the user your internal loop - only give them the final, concise answer

SATISFACTION INDICATOR:
- At the end of your response, add one of these tags on a new line:
  [SATISFIED] - if you have fully understood the user's goal and provided a complete answer
  [NEEDS_CLARIFICATION] - if you need more information from the user to proceed
- These tags are for internal processing only and will be removed before sending to the user

Available Tools:
- GitHub: Create/manage repos, files, branches, commits
- Netlify: Deploy projects, manage sites, set environment variables

Be proactive and helpful — like a developer friend on WhatsApp who gets things done.
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
            new_schema = {}
            for k, v in schema.items():
                if k not in ["additionalProperties", "$schema", "const"]:
                    new_schema[k] = self._clean_schema(v)
            return new_schema
        elif isinstance(schema, list):
            return [self._clean_schema(item) for item in schema]
        else:
            return schema
    
    def _filter_unset_parameters(self, arguments, schema):
        """Filter out unset/None parameters from arguments based on schema requirements"""
        if not schema or not isinstance(schema, dict):
            return arguments
        
        required_params = schema.get('required', [])
        properties = schema.get('properties', {})
        
        filtered_args = {}
        
        for key, value in arguments.items():
            if key in required_params:
                filtered_args[key] = value
            elif key in properties and value is not None and value != "":
                filtered_args[key] = value
        
        return filtered_args
    
    async def _get_mcp_tools(self, mcp_session):
        """Get MCP tools from a session"""
        try:
            logger.info("Retrieving MCP tools from session...")
            mcp_tools = await mcp_session.list_tools()
            logger.info(f"Retrieved {len(mcp_tools.tools)} MCP tools")
            
            declarations = []
            for tool in mcp_tools.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._clean_schema(tool.inputSchema),
                    "original_schema": tool.inputSchema
                }
                declarations.append(tool_info)
            
            return declarations
        except Exception as e:
            logger.error(f"Error getting MCP tools: {str(e)}")
            return []
    
    async def _execute_function_call(self, function_call, mcp_session, tool_schemas=None):
        """Execute a function call using MCP"""
        try:
            logger.info(f"Executing function call: {function_call.name}")
            
            original_args = dict(function_call.args)
            logger.info(f"Original function arguments: {original_args}")
            
            filtered_args = original_args
            if tool_schemas:
                tool_schema = None
                for schema in tool_schemas:
                    if schema.get("name") == function_call.name:
                        tool_schema = schema.get("original_schema")
                        break
                
                if tool_schema:
                    filtered_args = self._filter_unset_parameters(original_args, tool_schema)
                    logger.info(f"Filtered function arguments: {filtered_args}")
            
            result = await mcp_session.call_tool(
                function_call.name, 
                arguments=filtered_args
            )
            
            logger.info(f"Function call {function_call.name} executed successfully")
            
            if result.content and len(result.content) > 0:
                try:
                    result_data = json.loads(result.content[0].text)
                    return json.dumps(result_data, indent=2)
                except json.JSONDecodeError:
                    return result.content[0].text
            else:
                return "Function executed successfully but returned no content."
                        
        except Exception as e:
            logger.error(f"Error executing function call: {str(e)}")
            return f"Error executing function: {str(e)}"
        
    def generate_response(self, user_message, github_token=None, chat_history=None):
        """Generate a response using Gemini API with MCP integration"""
        try:
            logger.info("Generating AI response for user message")
            
            token_to_use = github_token or self.default_github_token
            
            return asyncio.run(self._generate_response_async(
                user_message, token_to_use, chat_history, self.system_prompt
            ))
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error. Please try again."
    
    async def _generate_response_async(self, user_message, github_token, chat_history, enhanced_system_prompt):
        """Async version of response generation with internal satisfaction loop"""
        try:
            # Prepare base conversation
            base_messages = [
                self._make_content("user", enhanced_system_prompt),
                self._make_content("model", "I understand. I'm your AI coding assistant ready to help you build apps and websites. How can I assist you today?")
            ]
            
            if chat_history:
                formatted = self._format_chat_history(chat_history)
                base_messages.extend(formatted)
                logger.info(f"Using chat history with {len(formatted)} previous messages")
            
            base_messages.append(self._make_content("user", user_message))
            
            # Load MCP tools
            tools = []
            all_tool_schemas = []
            tool_to_server_map = {}
            token_to_use = github_token or self.default_github_token
            
            github_declarations = []
            netlify_declarations = []
            
            # Load GitHub tools
            if token_to_use:
                try:
                    logger.info("Initializing MCP GitHub server connection...")
                    server_params = StdioServerParameters(
                        command="npx.cmd",
                        args=["-y", "@modelcontextprotocol/server-github"],
                        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token_to_use}
                    )
                    
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as github_session:
                            await github_session.initialize()
                            logger.info("GitHub MCP session initialized successfully")
                            
                            github_declarations = await self._get_mcp_tools(github_session)
                            for decl in github_declarations:
                                tool_to_server_map[decl["name"]] = "github"
                            all_tool_schemas.extend(github_declarations)
                            logger.info(f"Loaded {len(github_declarations)} GitHub MCP tools")
                except Exception as e:
                    logger.warning(f"Error loading GitHub MCP tools: {str(e)}")
            
            # Load Netlify tools
            if self.netlify_token:
                try:
                    logger.info("Initializing MCP Netlify server connection...")
                    server_params = StdioServerParameters(
                        command="npx.cmd",
                        args=["-y", "@netlify/mcp"],
                        env={"NETLIFY_API_KEY": self.netlify_token}
                    )
                    
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as netlify_session:
                            await netlify_session.initialize()
                            logger.info("Netlify MCP session initialized successfully")
                            
                            netlify_declarations = await self._get_mcp_tools(netlify_session)
                            for decl in netlify_declarations:
                                tool_to_server_map[decl["name"]] = "netlify"
                            all_tool_schemas.extend(netlify_declarations)
                            logger.info(f"Loaded {len(netlify_declarations)} Netlify MCP tools")
                except Exception as e:
                    logger.warning(f"Error loading Netlify MCP tools: {str(e)}")
            
            # Combine all tool declarations
            all_declarations = github_declarations + netlify_declarations
            
            if all_declarations:
                gemini_declarations = []
                for decl in all_declarations:
                    gemini_declarations.append({
                        "name": decl["name"],
                        "description": decl["description"],
                        "parameters": decl["parameters"]
                    })
                
                tools = [types.Tool(function_declarations=gemini_declarations)]
                logger.info(f"Total MCP tools loaded: {len(all_declarations)}")
            
            # Internal satisfaction loop
            max_iterations = 5
            iteration = 0
            conversation_messages = base_messages.copy()
            final_response = None
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Internal loop iteration {iteration}/{max_iterations}")
                
                config = types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=8192,
                )
                
                if tools:
                    config.tools = tools
                
                logger.info("Sending request to Gemini...")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=conversation_messages,
                    config=config,
                )
                
                # Extract response parts
                function_calls = []
                response_text = None
                is_satisfied = False
                
                if (response.candidates and 
                    len(response.candidates) > 0 and 
                    response.candidates[0].content.parts):
                    
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)
                            logger.info(f"Function call detected: {part.function_call.name}")
                        elif hasattr(part, 'text') and part.text:
                            response_text = part.text
                
                # Check satisfaction indicator
                if response_text:
                    if "[SATISFIED]" in response_text:
                        is_satisfied = True
                        response_text = response_text.replace("[SATISFIED]", "").strip()
                        logger.info("Agent is satisfied with the response")
                    elif "[NEEDS_CLARIFICATION]" in response_text:
                        is_satisfied = False
                        response_text = response_text.replace("[NEEDS_CLARIFICATION]", "").strip()
                        logger.info("Agent needs clarification, continuing loop")
                
                # Execute function calls if any
                if function_calls:
                    logger.info(f"Processing {len(function_calls)} function calls")
                    
                    try:
                        github_calls = []
                        netlify_calls = []
                        
                        for function_call in function_calls:
                            server_type = tool_to_server_map.get(function_call.name, "github")
                            if server_type == "netlify":
                                netlify_calls.append(function_call)
                            else:
                                github_calls.append(function_call)
                        
                        function_results = []
                        
                        # Execute GitHub calls
                        if github_calls and token_to_use:
                            logger.info(f"Executing {len(github_calls)} GitHub function call(s)")
                            server_params = StdioServerParameters(
                                command="npx.cmd",
                                args=["-y", "@modelcontextprotocol/server-github"],
                                env={"GITHUB_PERSONAL_ACCESS_TOKEN": token_to_use}
                            )
                            
                            async with stdio_client(server_params) as (read, write):
                                async with ClientSession(read, write) as github_session:
                                    await github_session.initialize()
                                    
                                    for function_call in github_calls:
                                        function_result = await self._execute_function_call(
                                            function_call, 
                                            github_session, 
                                            all_tool_schemas
                                        )
                                        function_results.append(f"{function_call.name}: {function_result}")
                        
                        # Execute Netlify calls
                        if netlify_calls and self.netlify_token:
                            logger.info(f"Executing {len(netlify_calls)} Netlify function call(s)")
                            server_params = StdioServerParameters(
                                command="npx.cmd",
                                args=["-y", "@netlify/mcp"],
                                env={"NETLIFY_API_KEY": self.netlify_token}
                            )
                            
                            async with stdio_client(server_params) as (read, write):
                                async with ClientSession(read, write) as netlify_session:
                                    await netlify_session.initialize()
                                    
                                    for function_call in netlify_calls:
                                        function_result = await self._execute_function_call(
                                            function_call, 
                                            netlify_session, 
                                            all_tool_schemas
                                        )
                                        function_results.append(f"{function_call.name}: {function_result}")
                        
                        all_results = "\n".join(function_results)
                        
                        # Add results to conversation for next iteration
                        conversation_messages.append(self._make_content("model", response_text or f"Executing {len(function_calls)} function(s)..."))
                        conversation_messages.append(self._make_content("user", f"Function results: {all_results}"))
                        
                        logger.info(f"Function calls completed, continuing internal loop")
                        
                    except Exception as function_error:
                        logger.error(f"Error executing function calls: {str(function_error)}")
                        final_response = f"I attempted to execute functions, but encountered an error: {str(function_error)}"
                        break
                
                elif response_text and is_satisfied:
                    # No function calls, satisfied - exit loop with final response
                    final_response = response_text
                    logger.info(f"Final response generated at iteration {iteration} (satisfied)")
                    break
                
                elif response_text and not is_satisfied:
                    # Agent needs clarification - add to conversation and continue
                    conversation_messages.append(self._make_content("model", response_text))
                    logger.info(f"Agent needs clarification at iteration {iteration}, continuing loop")
                
                else:
                    # No response or function calls
                    logger.warning(f"No response or function calls at iteration {iteration}")
                    if iteration >= max_iterations:
                        final_response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                        break
            
            # Fallback if no final response
            if not final_response:
                final_response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            # Format for WhatsApp
            final_response = format_code_for_whatsapp(final_response)
            final_response = truncate_message(final_response, max_length=1500)
            
            logger.info("Code formatting applied for WhatsApp display")
            logger.info(f"AI response generated successfully ({len(final_response)} chars)")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in async response generation: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}. Please try again with a simpler request."
