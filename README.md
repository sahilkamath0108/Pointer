# waWeb - WhatsApp AI Coding Assistant

A simple WhatsApp bot powered by Google Gemini AI that helps users build apps and websites through conversational coding assistance.

## Features

- 🤖 **AI-Powered Coding Help**: Get coding assistance through WhatsApp using Google Gemini AI
- 💬 **WhatsApp Integration**: Seamless interaction through Twilio WhatsApp API
- 🔗 **GitHub Token Support**: Store GitHub personal access tokens for enhanced project guidance
- 📚 **Chat History**: Maintains conversation context for better assistance
- 🌐 **REST API**: Test the AI assistant without WhatsApp using the HTTP API
- 🛠️ **Code Formatting**: Automatically formats code responses for WhatsApp display

## Quick Start

### Prerequisites

- Python 3.8+
- Twilio account with WhatsApp sandbox
- Google Gemini API key
- (Optional) GitHub personal access token

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd waWeb
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your values:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
   SECRET_KEY=your_secret_key_here
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## Usage

### WhatsApp Integration

1. **Set up Twilio WhatsApp Sandbox**
   - Go to Twilio Console > Messaging > Try it out > Send a WhatsApp message
   - Follow instructions to join the sandbox

2. **Configure Webhook**
   - Set webhook URL to: `https://your-domain.com/webhook`
   - Use ngrok for local development: `ngrok http 5000`

3. **Start Chatting**
   - Send messages to your Twilio WhatsApp number
   - The AI will respond with coding help and guidance

### GitHub Token (Optional)

To enable enhanced GitHub project guidance, send your personal access token:

```
TOKEN: your_github_personal_access_token_here
```

The AI will then provide more specific guidance for repository structure, code organization, and best practices.

### API Testing

Test the AI assistant without WhatsApp using the REST API:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Help me create a simple Python web app",
    "user_id": "test_user"
  }'
```

## API Endpoints

- `POST /webhook` - Twilio WhatsApp webhook
- `POST /api/chat` - Chat with AI assistant
- `GET /api/chat/history/<user_id>` - Get chat history
- `DELETE /api/chat/clear/<user_id>` - Clear chat history
- `GET /health` - Health check
- `GET /` - API information

## Project Structure

```
waWeb/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── services/
│   ├── ai_service.py     # Google Gemini AI integration
│   └── twilio_service.py # Twilio WhatsApp integration
├── utils/
│   ├── logger.py         # Logging configuration
│   └── code_formatter.py # Code formatting utilities
└── logs/                 # Application logs
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID | Yes |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token | Yes |
| `TWILIO_WHATSAPP_FROM` | Twilio WhatsApp number | Yes |
| `SECRET_KEY` | Flask secret key | No (auto-generated) |

### Getting API Keys

1. **Google Gemini API Key**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

2. **Twilio Credentials**
   - Sign up at [Twilio](https://www.twilio.com/)
   - Go to Console Dashboard
   - Copy Account SID and Auth Token
   - Set up WhatsApp sandbox for testing

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

1. **Using Gunicorn**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Using Docker** (optional)
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions:

1. Check the logs in the `logs/` directory
2. Ensure all environment variables are correctly set
3. Verify your API keys are valid
4. Check the health endpoint: `GET /health`

## Troubleshooting

### Common Issues

1. **"Gemini API key not found"**
   - Ensure `GEMINI_API_KEY` is set in your `.env` file

2. **"Twilio authentication failed"**
   - Verify `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are correct

3. **"WhatsApp messages not received"**
   - Check your webhook URL configuration in Twilio
   - Ensure your server is publicly accessible (use ngrok for local testing)

4. **"Internal server error"**
   - Check the application logs in the `logs/` directory
   - Verify all required environment variables are set 