-- Create analysis_tasks table
CREATE TABLE analysis_tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id VARCHAR UNIQUE NOT NULL,
  status VARCHAR NOT NULL DEFAULT 'pending',
  progress INTEGER DEFAULT 0,
  base_url VARCHAR NOT NULL,
  scraped_urls JSONB,
  combined_content TEXT,
  result JSONB,
  error TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  user_id UUID REFERENCES auth.users(id),
  business_name VARCHAR,
  total_pages_scraped INTEGER DEFAULT 0,
  crawl_delay_used FLOAT DEFAULT 1.0
);

-- Create business_analyses table
CREATE TABLE business_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id VARCHAR REFERENCES analysis_tasks(task_id),
  base_url VARCHAR NOT NULL,
  scraped_urls JSONB,
  company_overview TEXT,
  key_offerings_or_products JSONB,
  target_customer_segments JSONB,
  unique_selling_points JSONB,
  industry_and_market_trends JSONB,
  potential_business_challenges JSONB,
  opportunities_for_using_ai JSONB,
  recommended_ai_use_cases JSONB,
  data_requirements_and_risks JSONB,
  suggested_next_steps_for_ai_adoption JSONB,
  customer_journey_mapping TEXT,
  digital_maturity_assessment TEXT,
  technology_stack_overview JSONB,
  partnerships_and_alliances JSONB,
  sustainability_and_social_responsibility TEXT,
  financial_overview TEXT,
  actionable_recommendations JSONB,
  competitive_landscape JSONB,
  customer_testimonials JSONB,
  quantitative_opportunity_metrics JSONB,
  content_inventory JSONB,
  ai_maturity_level VARCHAR,
  data_sources_reviewed JSONB,
  business_stage VARCHAR,
  branding_tone VARCHAR,
  visual_opportunities JSONB,
  team_ai_readiness TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_analysis_tasks_status ON analysis_tasks(status);
CREATE INDEX idx_analysis_tasks_created_at ON analysis_tasks(created_at);
CREATE INDEX idx_analysis_tasks_base_url ON analysis_tasks(base_url);
CREATE INDEX idx_business_analyses_task_id ON business_analyses(task_id);
CREATE INDEX idx_business_analyses_base_url ON business_analyses(base_url);
CREATE INDEX idx_business_analyses_created_at ON business_analyses(created_at);

-- Enable Row Level Security (RLS) - optional for security
ALTER TABLE analysis_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_analyses ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (adjust based on your auth needs)
CREATE POLICY "Allow public read access to analysis_tasks" ON analysis_tasks
  FOR SELECT USING (true);

CREATE POLICY "Allow public insert access to analysis_tasks" ON analysis_tasks
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public update access to analysis_tasks" ON analysis_tasks
  FOR UPDATE USING (true);

CREATE POLICY "Allow public read access to business_analyses" ON business_analyses
  FOR SELECT USING (true);

CREATE POLICY "Allow public insert access to business_analyses" ON business_analyses
  FOR INSERT WITH CHECK (true); 