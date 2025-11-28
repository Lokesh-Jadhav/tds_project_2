# LLM Analysis Quiz Solver

This project implements an automated quiz-solving API that can:
- Render JavaScript websites using Playwright
- Download and parse CSV, Excel, JSON, PDF files
- Analyze tables using Pandas
- Extract numeric data
- Perform aggregation (sum, mean, min, max)
- Generate charts using Matplotlib
- Submit results automatically to required endpoints

---

## API Endpoint

POST `/quiz`

### Request JSON:
```json
{
  "email": "your@email.com",
  "secret": "your_secret",
  "url": "quiz_url"
}
