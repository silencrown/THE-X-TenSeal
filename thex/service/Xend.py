import enum

class Protocol(enum.Enum):
    GRPC = "grpc"
    TCP = "tcp"
    HTTP2 = "http2"

class Xend:
    """
    This is an example of a function that does something.

    Parameters

    ip: The IP address of the server.
    port: The port on which the server is running.
    protocol: The protocol to use. Either Protocol.GRPC or Protocol.HTTP.
    cache_location: The location where cached data is stored on the local machine.
    minio: A dictionary containing the endpoint, access key, secret key, and bucket name for the MinIO server.

    Usage:
        xend = Xend(
            ip="192.168.0.1",
            port=8000,
            protocol=Protocol.GRPC,
            cache_location="/tmp/cache",
            minio={
                "endpoint": "127.0.0.1:9000",
                "access_key": "your_access_key",
                "secret_key": "your_secret_key",
                "bucket_name": "your_bucket_name",
            },
        )
    """
    def __init__(self, ip: str, port: int, protocol: Protocol, cache_location: str, minio: dict):
        self.ip = ip
        self.port = port
        self.protocol = protocol
        self.cache_location = cache_location
        self.minio = minio

    def __repr__(self):
        return f"Xend(ip={self.ip}, port={self.port}, protocol={self.protocol.value}, cache_location={self.cache_location}, minio={self.minio})"

    def __str__(self):
        return f"Xend instance with IP: {self.ip}, Port: {self.port}, Protocol: {self.protocol.value}, Cache Location: {self.cache_location}, MinIO Configuration: {self.minio}"

    def connect(self):
        # Implement the connection logic based on the selected protocol
        if self.protocol == Protocol.GRPC:
            # Connect using gRPC
            pass
        elif self.protocol == Protocol.TCP:
            # Connect using TCP
            pass
        elif self.protocol == Protocol.HTTP2:
            # Connect using HTTP/2
            pass

    def send_data(self, data):
        # Implement the data sending logic based on the selected protocol
        if self.protocol == Protocol.GRPC:
            # Send data using gRPC
            pass
        elif self.protocol == Protocol.TCP:
            # Send data using TCP
            pass
        elif self.protocol == Protocol.HTTP2:
            # Send data using HTTP/2
            pass

    def receive_data(self):
        # Implement the data receiving logic based on the selected protocol
        if self.protocol == Protocol.GRPC:
            # Receive data using gRPC
            pass
        elif self.protocol == Protocol.TCP:
            # Receive data using TCP
            pass
        elif self.protocol == Protocol.HTTP2:
            # Receive data using HTTP/2
            pass

    def disconnect(self):
        # Implement the disconnection logic based on the selected protocol
        if self.protocol == Protocol.GRPC:
            # Disconnect using gRPC
            pass
        elif self.protocol == Protocol.TCP:
            # Disconnect using TCP
            pass
        elif self.protocol == Protocol.HTTP2:
            # Disconnect using HTTP/2
            pass


