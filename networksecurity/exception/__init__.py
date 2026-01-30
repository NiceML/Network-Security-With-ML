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
        exc_info = error_details.exc_info()
        
        if exc_info and exc_info[2] is not None:
            exc_tb = exc_info[2]
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = 0
            self.file_name = "unknown"
        
    def __str__(self):
        """
        Format the error message
        """
        return f"Error occurred in python script name [{self.file_name}] line number [{self.lineno}] error message [{self.error_message}]"
