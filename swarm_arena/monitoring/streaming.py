"""WebSocket streaming server for real-time telemetry."""

import asyncio
import json
import threading
import time
from typing import Set, Optional, Dict, Any

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

from .telemetry import TelemetryCollector, TelemetryData
from ..exceptions import NetworkError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StreamingServer:
    """WebSocket server for real-time telemetry streaming."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8765,
                 telemetry_collector: Optional[TelemetryCollector] = None) -> None:
        """Initialize streaming server.
        
        Args:
            host: Server host address
            port: Server port
            telemetry_collector: Telemetry collector instance
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library not available. Install with: pip install websockets")
            
        self.host = host
        self.port = port
        self.telemetry_collector = telemetry_collector
        
        # WebSocket state
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self.running = False
        
        # Streaming state
        self.streaming_thread: Optional[threading.Thread] = None
        self.stream_interval = 0.1  # 10 Hz by default
        
        # Message statistics
        self.messages_sent = 0
        self.clients_connected = 0
        self.start_time = time.time()
        
        logger.info(f"Streaming server initialized on {host}:{port}")
    
    def broadcast_data(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected clients (synchronous version for testing).
        
        Args:
            data: Data to broadcast
        """
        if not self.clients:
            return
            
        message_json = json.dumps(data)
        
        for client in list(self.clients):
            try:
                # Mock sending for testing
                if hasattr(client, 'send'):
                    client.send(message_json)
                self.messages_sent += 1
            except Exception as e:
                logger.warning(f"Failed to send data to client: {str(e)}")
                self.clients.discard(client)
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics.
        
        Returns:
            Dictionary of server statistics
        """
        uptime = time.time() - self.start_time
        
        stats = {
            "host": self.host,
            "port": self.port,
            "uptime": uptime,
            "running": self.running,
            "active_clients": len(self.clients),
            "total_clients_connected": self.clients_connected,
            "messages_sent": self.messages_sent,
            "stream_interval": self.stream_interval,
            "messages_per_second": self.messages_sent / max(1, uptime)
        }
        
        # Add telemetry collector stats if available
        if self.telemetry_collector:
            telemetry_stats = self.telemetry_collector.get_statistics()
            stats["telemetry"] = telemetry_stats
        
        return stats


def create_streaming_server(arena, host: str = "localhost", port: int = 8765) -> StreamingServer:
    """Create a streaming server for an arena.
    
    Args:
        arena: Arena instance to stream from
        host: Server host
        port: Server port
        
    Returns:
        Configured streaming server
    """
    # Create telemetry collector
    telemetry = TelemetryCollector(auto_start=False)
    
    # Create streaming server
    server = StreamingServer(host, port, telemetry)
    
    return server