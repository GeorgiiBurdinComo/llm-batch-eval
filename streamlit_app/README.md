# Streamlit Model Accuracy Dashboard

A public, shareable Streamlit application that fetches and visualizes model accuracy metrics directly from the Langfuse REST API.

This app allows you to share your benchmark results dynamically without requiring stakeholders to have a Langfuse account.

## Running Locally

1. **Install dependencies:**
   Make sure to install the required packages.
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   The app reads your Langfuse credentials from the environment.
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
   export LANGFUSE_SECRET_KEY="sk-lf-..."
   export LANGFUSE_HOST="https://cloud.langfuse.com" # Optional, defaults to Langfuse Cloud
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py --server.runOnSave true
   ```
   This will open the dashboard in your default web browser (usually at `http://localhost:8501`). The `--server.runOnSave true` flag ensures the app automatically refreshes whenever you save changes to `app.py`.

## Deploying to Streamlit Community Cloud (Free)

To make this dashboard fully public on the internet:

1. Push this code to a public or private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Deploy a new app and select the branch and file path (`streamlit_app/app.py`).
4. **Important:** Before clicking "Deploy", click on "Advanced settings..." and add your Langfuse API keys to the **Secrets** section:
   ```toml
   LANGFUSE_PUBLIC_KEY = "pk-lf-..."
   LANGFUSE_SECRET_KEY = "sk-lf-..."
   LANGFUSE_HOST = "https://cloud.langfuse.com"
   ```
5. Click **Deploy**. Your dashboard will be live and shareable via the Streamlit app URL!
