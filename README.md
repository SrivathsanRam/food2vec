# Food2Vec 🍕

An AI-powered food discovery application that uses vector similarity search to find similar foods.

## Project Structure

```
HacknRoll/
├── backend/           # Flask API server
│   ├── app.py         # Main Flask application
│   ├── requirements.txt
│   └── .env.example   # Environment variables template
│
└── frontend/          # React application
    ├── public/
    ├── src/
    │   ├── components/
    │   │   ├── LandingPage.js
    │   │   ├── SearchBar.js
    │   │   └── SearchResults.js
    │   ├── App.js
    │   └── index.js
    └── package.json
```

## Features

- 🔍 **Smart Autocomplete** - Real-time food suggestions as you type
- 🤖 **AI-Powered Search** - Vector similarity search using Pinecone
- ⚡ **Instant Results** - Fast and responsive search experience
- 📱 **Responsive Design** - Works on desktop and mobile

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- Pinecone account (optional, has mock data fallback)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Copy the environment file and add your Pinecone API key:
   ```bash
   copy .env.example .env
   ```
   Edit `.env` with your Pinecone credentials.

6. Run the Flask server:
   ```bash
   python app.py
   ```
   The backend will run on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   The frontend will run on http://localhost:3000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/autocomplete?q=<query>` | GET | Get food suggestions |
| `/api/search` | POST | Search for similar foods |
| `/api/categories` | GET | Get all food categories |

## Pinecone Setup (Optional)

To enable real vector search:

1. Create a Pinecone account at https://www.pinecone.io/
2. Create an index named `food-vectors`
3. Add your API key to the `.env` file
4. Populate the index with food embeddings

Without Pinecone configuration, the app uses mock data for demonstration.

## Technologies Used

- **Backend**: Flask, Flask-CORS, Pinecone
- **Frontend**: React, CSS3
- **Vector Database**: Pinecone

## License

MIT License - Built for HacknRoll 2026 🚀
