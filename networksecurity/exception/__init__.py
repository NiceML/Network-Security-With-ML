import sys

class NetworkSecurityException(Exception):
    """
    Custom exception class for Network Security module
    """
    
    def __init__(self, error_message, error_details: sys):
        """
        Initialize the exception with error message and details
        
        Args:
            error_message: The error message
            error_details: System error details
        """
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        
    def __str__(self):
        """
        Format the error message
        """
        return f"Error occurred in python script name [{self.file_name}] line number [{self.lineno}] error message [{self.error_message}]"
