You maintain a running summary of a chat session so future turns have context without bloating the prompt.

Update the existing summary using the most recent turn. Keep it under 6 short bullets. Preserve user goals, named entities, and any constraints the user has stated.

EXISTING SUMMARY:
{existing_summary}

NEW TURN:
USER: {user_message}
ASSISTANT: {assistant_message}

UPDATED SUMMARY:
