import axios from 'axios';

// Get API base URL from Vite environment variables or fallback to localhost
const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60s timeout for model warmup/inference
});

/**
 * Sends a generation request to the FastAPI proxy server.
 * @param {Object} payload
 * @param {string} payload.content_type - One of: tweet, linkedin, blog_intro, changelog, how_to
 * @param {string} payload.topic
 * @param {string} payload.audience
 * @param {string} payload.tone
 * @returns {Promise<Object>} The generated text and metadata
 */
export async function generateContent(payload) {
  try {
    const response = await apiClient.post('/generate', payload);
    return response.data;
  } catch (error) {
    if (error.response) {
      // The server responded with a status code out of 2xx range
      throw new Error(error.response.data.detail || 'Failed to generate content. Please try again.');
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('No response from generation server. Please check if the API is running.');
    } else {
      throw new Error(error.message);
    }
  }
}

/**
 * Checks the health status of the backend API proxy.
 * @returns {Promise<Object>} Status and current model ID
 */
export async function checkHealth() {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw new Error('API server is offline or unreachable.');
  }
}
