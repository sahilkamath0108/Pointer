import re
from utils.logger import logger

def format_code_for_whatsapp(text):
    """
    Format code snippets in AI responses for WhatsApp
    WhatsApp has limited markdown support, so this ensures code is readable
    """
    try:
        # Find code blocks with language specification
        pattern = r"```(\w+)\n([\s\S]*?)\n```"
        matches = re.findall(pattern, text)
        
        for lang, code in matches:
            # Format replacement with language and clean indentation
            replacement = f"*{lang.upper()} CODE:*\n```\n{code}\n```"
            text = text.replace(f"```{lang}\n{code}\n```", replacement)
        
        # Find generic code blocks
        pattern = r"```\n([\s\S]*?)\n```"
        matches = re.findall(pattern, text)
        
        for code in matches:
            # Format replacement with clean indentation
            replacement = f"*CODE:*\n```\n{code}\n```"
            text = text.replace(f"```\n{code}\n```", replacement)
            
        logger.info("Code formatting applied for WhatsApp display")
        return text
    except Exception as e:
        logger.error(f"Error formatting code: {str(e)}")
        return text  # Return original text if formatting fails

def truncate_message(text, max_length=1500):
    """
    Truncate message to fit WhatsApp character limit
    """
    if len(text) <= max_length:
        return text
        
    # Try to find a good break point
    truncated = text[:max_length-3]
    
    # Try to end at a paragraph
    last_newline = truncated.rfind('\n\n')
    if last_newline > max_length * 0.8:  # If we found a paragraph break in the last 20%
        truncated = truncated[:last_newline]
    
    logger.info(f"Message truncated from {len(text)} to {len(truncated)+3} chars")
    return truncated + "..." 