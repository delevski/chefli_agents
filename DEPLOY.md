# Fly.io Deployment Guide

## Prerequisites

1. Sign up for a free Fly.io account: https://fly.io (no payment method required)
2. Install Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

## Deployment Steps

### 1. Login to Fly.io
```bash
fly auth signup
# or if you already have an account:
fly auth login
```

### 2. Launch Your App
```bash
fly launch
```

This will:
- Create a `fly.toml` file (already created)
- Ask you to name your app (or use default)
- Ask about regions (choose closest to you)
- Ask about Postgres/Redis (say no for this app)
- Deploy your app

### 3. Set Environment Variables (Secrets)

Set your API keys as secrets:
```bash
fly secrets set OPENAI_API_KEY=your_openai_api_key_here
fly secrets set LANGCHAIN_API_KEY=your_langsmith_api_key_here
fly secrets set LANGCHAIN_PROJECT=chefli-agents
fly secrets set LANGCHAIN_TRACING_V2=true
fly secrets set LLM_PROVIDER=openai
```

### 4. Deploy
```bash
fly deploy
```

### 5. Check Your App
```bash
# View logs
fly logs

# Open your app in browser
fly open

# Check status
fly status
```

## Useful Commands

```bash
# View logs in real-time
fly logs

# SSH into your app
fly ssh console

# Scale your app (if needed)
fly scale count 1

# View app info
fly info

# List all apps
fly apps list
```

## Your App URL

After deployment, your app will be available at:
```
https://chefli-agents.fly.dev
```

## Troubleshooting

### If deployment fails:
1. Check logs: `fly logs`
2. Verify Dockerfile is correct
3. Check environment variables: `fly secrets list`
4. Test locally: `docker build -t chefli-agents .`

### If app times out:
- LLM calls can take 30-60 seconds
- Fly.io default timeout is 60s (should be enough)
- If needed, increase timeout in `fly.toml`

### If you need to update:
```bash
# Make changes to your code
git add .
git commit -m "Update app"

# Redeploy
fly deploy
```

## Free Tier Limits

- 3 shared VMs
- 160GB outbound data transfer/month
- Apps auto-sleep after inactivity (wake on first request)

## Cost

- **Free tier**: No cost, no payment method required
- Apps sleep when not in use (first request may be slow)
- Perfect for development and testing
