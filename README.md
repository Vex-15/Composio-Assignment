# Composio App Research Pipeline

AI-powered research pipeline built for the Composio Product Intern assignment.

## Clone the Repository

```bash
git clone https://github.com/Vex-15/Composio-Assignment.git
cd Composio-Assignment
```

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configure

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_API_KEY
COMPOSIO_API_KEY=YOUR_API_KEY
SERPER_API_KEY=YOUR_API_KEY
```

---

## Run the Research Pipeline

```bash
python -m agents.research
python -m agents.verify
python -m agents.insights
python -m agents.report
```

---

## View the Report

Open:

```
reports/case-study.html
```

or serve it locally:

```bash
cd reports
python -m http.server 8080
```

Then visit:

```
http://localhost:8080/case-study.html
```

---

## Preview

<img width="1918" height="889" alt="Dashboard" src="https://github.com/user-attachments/assets/0ab69509-9398-422d-83a4-e42ce03d1bf1" />

<img width="1916" height="906" alt="Case Study" src="https://github.com/user-attachments/assets/99e65fdf-1edf-4344-beee-66d7270025e6" />

<img width="1828" height="923" alt="Insights" src="https://github.com/user-attachments/assets/033b7e3f-b55f-4eba-996a-c6b1f740a735" />

<img width="1816" height="894" alt="Report" src="https://github.com/user-attachments/assets/e0759aa2-f270-437e-8828-471f437eb7d1" />

---

## Tech Stack

- Python
- Google Gemini API
- Composio SDK
- HTTPX
- BeautifulSoup
- asyncio
- Tenacity
- tqdm
