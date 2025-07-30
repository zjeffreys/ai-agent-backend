# Supabase Setup for Task-Based Business Analysis

## 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Sign up/Login and create a new project
3. Choose a project name and database password
4. Wait for the project to be created

## 2. Get Your Supabase Credentials

1. In your Supabase dashboard, go to **Settings** → **API**
2. Copy the following values:
   - **Project URL** (looks like: `https://your-project-id.supabase.co`)
   - **anon public** key (starts with `eyJ...`)

## 3. Set Up Environment Variables

Add these to your `.env` file:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
```

## 4. Create Database Tables

1. In your Supabase dashboard, go to **SQL Editor**
2. Copy and paste the contents of `supabase_schema.sql`
3. Click **Run** to execute the SQL

This will create:
- `analysis_tasks` table - stores task status and progress
- `business_analyses` table - stores completed analysis results
- Indexes for better performance
- Row Level Security policies

## 5. Test the Setup

1. Install the new dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start your FastAPI server:
   ```bash
   python main.py
   ```

3. Test the new endpoints:
   ```bash
   # Create a task
   curl -X POST "http://localhost:8000/analyze-business-task" \
     -H "Content-Type: application/json" \
     -d '{
       "combined_content": "Test content...",
       "base_url": "https://example.com"
     }'

   # Check task status (replace TASK_ID with the returned task_id)
   curl "http://localhost:8000/task-status/TASK_ID"
   ```

## 6. New API Endpoints

### Task Management
- `POST /analyze-business-task` - Create a new analysis task
- `GET /task-status/{task_id}` - Get task status and progress
- `GET /task-details/{task_id}` - Get detailed task information

### Analysis Results
- `GET /analyses` - Get all completed analyses
- `GET /analyses/{analysis_id}` - Get specific analysis
- `GET /analyses/by-url/{base_url}` - Get analyses for a specific website

## 7. Frontend Integration

```javascript
// Create a new analysis task
const createTask = async (scrapedData) => {
  const response = await fetch('/analyze-business-task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      combined_content: scrapedData.combinedContent,
      base_url: scrapedData.baseUrl
    })
  });
  
  const { task_id } = await response.json();
  return task_id;
};

// Poll for task status
const pollTaskStatus = async (taskId) => {
  const response = await fetch(`/task-status/${taskId}`);
  return response.json();
};

// Get all analyses for a website
const getAnalysesForWebsite = async (baseUrl) => {
  const response = await fetch(`/analyses/by-url/${encodeURIComponent(baseUrl)}`);
  return response.json();
};
```

## 8. Data Flow

1. **Frontend scrapes website** → Gets combined content and base URL
2. **Frontend calls `/analyze-business-task`** → Creates task in Supabase
3. **Backend processes task** → Updates progress in database
4. **Frontend polls `/task-status/{task_id}`** → Shows progress to user
5. **Analysis completes** → Results saved to `business_analyses` table
6. **Frontend gets results** → Displays analysis to user

## 9. Benefits

- ✅ **No data loss** - Everything saved to Supabase
- ✅ **Progress tracking** - Real-time status updates
- ✅ **Historical data** - All analyses stored permanently
- ✅ **URL tracking** - Know exactly what was analyzed
- ✅ **Scalable** - Database handles concurrent requests
- ✅ **Backup** - Supabase provides automatic backups

## 10. Troubleshooting

### Common Issues:

1. **"Failed to create task"** - Check your Supabase credentials in `.env`
2. **"Task not found"** - Verify the task_id exists in the database
3. **"Failed to fetch analyses"** - Check if tables were created correctly

### Debug Steps:

1. Check Supabase dashboard → **Table Editor** → Verify tables exist
2. Check **Logs** in Supabase dashboard for SQL errors
3. Verify environment variables are loaded correctly
4. Test database connection with a simple query

## 11. Next Steps

After setup, you can:
- Add user authentication
- Implement rate limiting
- Add analysis caching
- Create dashboard for viewing all analyses
- Add export functionality
- Implement analysis comparison features 