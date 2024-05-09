from uvicorn import Config, Server

'''
This is a wrapper for uvicorn server. 
It is used to run the uvicorn server from the main.py file.
Only used for local development to run and debug the app from the IDE.
'''


class UvicornWrapper:
    def __init__(self, app, host="0.0.0.0", port=10000, log_level="info"):
        self.config = Config(app, host=host, port=port, log_level=log_level)
        self.server = Server(config=self.config)

    def run(self):
        return self.server.run()

    def install_signal_handlers(self):
        self.server.install_signal_handlers()
