import os
import json
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from openai import OpenAI
from .db import get_connection

API_BASE_URL = os.environ.get("TOKEN_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SLACK_TOKEN = os.environ.get("SLACK_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")

client = OpenAI(api_key=OPENAI_API_KEY)

slack = WebClient(token=SLACK_TOKEN)

def create_ticket(description: str) -> str:
    """
    Creates a new support ticket in the internal system via a POST request.
    Args:
        description (str): Detailed description of the problem.
    Returns:
        str: A JSON string confirming the ticket creation details, including the new ticket ID.
    """
    print("creating new ticket for user")
    endpoint = f"{API_BASE_URL}/ticket"
    payload = {
        "session": "Chat Session",
        "description": description
    }
    print("payload", payload)
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        print(response.json())
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"ERROR: Failed to create ticket. Reason: {e}"
    
def get_ticket_details(ticket_id: str) -> str:
    """
    Retrieves detailed information for a single specific ticket via a GET request.
    Args:
        ticket_id (str): The unique identifier for the ticket.
    Returns:
        str: A JSON string of the ticket details or an error message.
    """
    endpoint = f"{API_BASE_URL}/ticket/{ticket_id}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        print(response.json())
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"ERROR: Failed to retrieve ticket details. Reason: {e}"

def post_message(message_text: str) -> str:
    """
    Posts a message to a designated Slack Channel.
    Args:
        message_text: The content of the message to post.
    Returns:
        str: A confirmation message including the channel and message status.
    """
    try:
        response = slack.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=message_text
        )
        print("Message sent:", response["ts"])
        status = f"Message posted to #support-team channel successfully"
        return status
    except SlackApiError as e:
        print("Error:", e.response["error"])

def create_ticket_and_notify_slack(description: str) -> str:
    """
    Creates a new support ticket and automatically notifies the Slack channel.
    This tool combines two actions sequentially.
    Args:
        description: Detailed description of the problem.
    Returns:
        str: The final outcome of both operations.
    """
    # step 1: Create the ticket
    ticket_id = create_ticket(description=description)

    # step 2: Use the output of step 1 as input for the next action
    notification_message = f"New Ticket Created: {ticket_id}. Detail: {description}"
    slack_status = post_message(notification_message)

    return f"Workflow completed. {slack_status}. Ticket ID: {ticket_id}"

def generate_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using an external model."""
    response = client.embeddings.create(
        model="text-embedding-3-small", # Or another non-Vertex model
        input=[text]
    )
    return response.data[0].embedding

def search_knowledge_base(query: str, top_k: int = 5):
    query_embedding = generate_embedding(query)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM match_documents(%s, %s)",
                (query_embedding, top_k)
            )
            return cur.fetchall()
    finally:
        conn.close()