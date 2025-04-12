import os
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables, Gemini features may not work properly")

# Configure the Gemini API client
genai.configure(api_key=API_KEY)

# Store chat sessions
chat_history = {}

def get_treatment_recommendation(disease_name):
    """
    Get treatment recommendations for a plant disease using Google Gemini
    
    Args:
        disease_name (str): The name of the plant disease
    
    Returns:
        str: Treatment recommendations
    """
    try:
        # Clean up disease name (replace underscores with spaces)
        disease = disease_name.replace('_', ' ')
        
        # Prepare the prompt for Gemini
        prompt = f"""
        You are a plant disease expert. Provide treatment recommendations for plants affected by {disease}.
        
        Follow this structure in your response:
        1. Brief description of the disease
        2. Symptoms
        3. Treatment recommendations (organic and chemical options)
        4. Prevention tips
        
        Keep your response informative but concise (less than 500 words).
        """
        
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Extract and return the text
        if response:
            return response.text
        else:
            return "Unable to generate treatment recommendations at this time."
    
    except Exception as e:
        logger.error(f"Error generating treatment recommendations: {str(e)}")
        
        # Fallback response if Gemini API fails
        return f"""
        # Treatment Recommendations for {disease_name.replace('_', ' ')}

        ## Description
        This is a common plant disease that affects crops and ornamental plants.

        ## Symptoms
        - Discoloration of leaves
        - Spots or lesions
        - Wilting or stunted growth

        ## Treatment
        - Remove affected plant parts
        - Apply appropriate fungicide or insecticide
        - Ensure proper plant nutrition

        ## Prevention
        - Rotate crops
        - Use disease-resistant varieties
        - Maintain good air circulation
        - Water at the base of plants to keep foliage dry

        *Note: These are general recommendations. For specific treatment, please consult with a local agricultural extension service.*
        """

def initialize_chat(session_id):
    """
    Initialize a new chat session with Gemini
    
    Args:
        session_id (str): Unique identifier for the chat session
    
    Returns:
        None
    """
    try:
        if not API_KEY:
            logger.warning("Cannot initialize chat: GEMINI_API_KEY not found")
            return False
            
        # Initialize the Gemini model for chat
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Initialize chat session with system prompt (Gemini doesn't support system messages in the same way)
        # So we'll start with an initial user/model exchange to set context
        chat = model.start_chat()
        
        # Prime the chat with initial context
        system_prompt = """You are PlantCare AI, a helpful agriculture assistant that specializes in plant disease diagnosis and treatment. 
        Your goal is to assist farmers, gardeners, and agricultural professionals with plant health issues.
        
        When responding to questions:
        1. Provide accurate, actionable advice for plant care
        2. Be concise but thorough in your explanations
        3. Remember that prevention is as important as treatment
        4. Consider both organic and conventional treatment options
        
        Focus on being helpful, educational, and practical in your responses."""
        
        # Send a hidden "system" message to set the context
        response = chat.send_message("You are a plant disease expert assistant. Please acknowledge your role.")
        
        # Store the chat session
        chat_history[session_id] = chat
        
        logger.info(f"Chat session initialized for session_id: {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}")
        return False

def chat_with_gemini(session_id, user_message):
    """
    Send a message to Gemini and get a response using a persistent chat session
    
    Args:
        session_id (str): The chat session identifier
        user_message (str): The user's message
        
    Returns:
        str: Gemini's response
    """
    try:
        if not API_KEY:
            return "Sorry, I cannot respond at the moment because the API key is missing. Please contact the administrator."
        
        # Check if session exists, if not create one
        if session_id not in chat_history:
            success = initialize_chat(session_id)
            if not success:
                return "Failed to initialize chat session. Please try again later."
        
        # Get the chat session
        chat = chat_history[session_id]
        
        # Send message and get response
        response = chat.send_message(user_message)
        
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "I couldn't generate a response. Please try again."
            
    except Exception as e:
        logger.error(f"Error in chat response: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request. Please try again later."
