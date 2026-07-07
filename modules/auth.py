"""Authentication helpers exposed as a dedicated module."""

import modules.core as core


def sanitize_email(email: str) -> str:
    """Replace characters not suitable for GCS object names."""
    return core.sanitize_email(email)


def authenticate_user_function(f):
    """Authenticate requests using the shared core decorator."""
    return core.authenticate_user_function(f)


def get_user_email_from_token():
    """Extract the authenticated user's email from the shared core helper."""
    return core.get_user_email_from_token()
