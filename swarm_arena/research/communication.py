"""Communication protocols and message passing for agent interactions."""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time


class MessageType(Enum):
    """Types of messages agents can send."""
    PROPOSAL = "proposal"
    ACCEPT = "accept"
    REJECT = "reject"
    COUNTER = "counter"
    INFO = "info"
    REQUEST = "request"
    BROADCAST = "broadcast"
    EMERGENCY = "emergency"


@dataclass
class Message:
    """Represents a message between agents."""
    
    sender_id: int
    receiver_id: Optional[int]  # None for broadcast messages
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time to live in simulation steps
    priority: int = 1  # 1 = low, 5 = high priority
    
    def encode(self) -> bytes:
        """Encode message for transmission."""
        data = {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "priority": self.priority
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def decode(cls, data: bytes) -> 'Message':
        """Decode message from transmission."""
        parsed = json.loads(data.decode('utf-8'))
        return cls(
            sender_id=parsed["sender_id"],
            receiver_id=parsed["receiver_id"],
            message_type=MessageType(parsed["message_type"]),
            content=parsed["content"],
            timestamp=parsed["timestamp"],
            ttl=parsed["ttl"],
            priority=parsed["priority"]
        )


class MessageChannel:
    """Manages message passing between agents with constraints."""
    
    def __init__(self, 
                 max_range: float = 100.0,
                 bandwidth_limit: int = 10,
                 noise_probability: float = 0.0,
                 latency_steps: int = 0):
        """Initialize message channel.
        
        Args:
            max_range: Maximum communication range
            bandwidth_limit: Max messages per agent per step
            noise_probability: Probability of message corruption
            latency_steps: Delay in message delivery
        """
        self.max_range = max_range
        self.bandwidth_limit = bandwidth_limit
        self.noise_probability = noise_probability
        self.latency_steps = latency_steps
        
        # Message queues
        self.pending_messages: List[Message] = []
        self.delayed_messages: List[tuple] = []  # (message, delivery_step)
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
            "messages_corrupted": 0
        }
    
    def send_message(self, message: Message, agent_positions: Dict[int, np.ndarray],
                    current_step: int) -> bool:
        """Send a message through the channel.
        
        Args:
            message: Message to send
            agent_positions: Current positions of all agents
            current_step: Current simulation step
            
        Returns:
            True if message was accepted for transmission
        """
        # Check if sender exists
        if message.sender_id not in agent_positions:
            return False
        
        # Check bandwidth limit for sender
        sender_message_count = sum(
            1 for m in self.pending_messages 
            if m.sender_id == message.sender_id
        )
        
        if sender_message_count >= self.bandwidth_limit:
            self.stats["messages_dropped"] += 1
            return False
        
        # Check range constraints for targeted messages
        if message.receiver_id is not None:
            if message.receiver_id not in agent_positions:
                self.stats["messages_dropped"] += 1
                return False
            
            sender_pos = agent_positions[message.sender_id]
            receiver_pos = agent_positions[message.receiver_id]
            distance = np.linalg.norm(sender_pos - receiver_pos)
            
            if distance > self.max_range:
                self.stats["messages_dropped"] += 1
                return False
        
        # Apply noise corruption
        if np.random.random() < self.noise_probability:
            message = self._corrupt_message(message)
            self.stats["messages_corrupted"] += 1
        
        # Add latency delay
        if self.latency_steps > 0:
            delivery_step = current_step + self.latency_steps
            self.delayed_messages.append((message, delivery_step))
        else:
            self.pending_messages.append(message)
        
        self.stats["messages_sent"] += 1
        return True
    
    def get_messages_for_agent(self, agent_id: int, current_step: int) -> List[Message]:
        """Get all pending messages for a specific agent.
        
        Args:
            agent_id: Agent to get messages for
            current_step: Current simulation step
            
        Returns:
            List of messages for the agent
        """
        # Process delayed messages
        ready_messages = []
        remaining_delayed = []
        
        for message, delivery_step in self.delayed_messages:
            if delivery_step <= current_step:
                ready_messages.append(message)
            else:
                remaining_delayed.append((message, delivery_step))
        
        self.delayed_messages = remaining_delayed
        self.pending_messages.extend(ready_messages)
        
        # Filter messages for this agent
        agent_messages = []
        remaining_messages = []
        
        for message in self.pending_messages:
            # Decrement TTL
            message.ttl -= 1
            
            if message.ttl <= 0:
                # Message expired
                continue
            
            # Check if message is for this agent
            if (message.receiver_id == agent_id or 
                message.receiver_id is None):  # Broadcast message
                agent_messages.append(message)
                self.stats["messages_delivered"] += 1
            else:
                remaining_messages.append(message)
        
        self.pending_messages = remaining_messages
        
        # Sort by priority
        agent_messages.sort(key=lambda m: m.priority, reverse=True)
        
        return agent_messages
    
    def _corrupt_message(self, message: Message) -> Message:
        """Apply noise corruption to a message."""
        corrupted_content = message.content.copy()
        
        # Randomly corrupt numeric values
        for key, value in corrupted_content.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, 0.1) * value
                corrupted_content[key] = value + noise
            elif isinstance(value, str):
                # Randomly flip characters
                if len(value) > 0 and np.random.random() < 0.1:
                    char_list = list(value)
                    flip_idx = np.random.randint(len(char_list))
                    char_list[flip_idx] = chr(ord(char_list[flip_idx]) ^ 1)
                    corrupted_content[key] = ''.join(char_list)
        
        # Create new message with corrupted content
        return Message(
            sender_id=message.sender_id,
            receiver_id=message.receiver_id,
            message_type=message.message_type,
            content=corrupted_content,
            timestamp=message.timestamp,
            ttl=message.ttl,
            priority=message.priority
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.stats.copy()
    
    def reset(self) -> None:
        """Reset the message channel."""
        self.pending_messages.clear()
        self.delayed_messages.clear()
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
            "messages_corrupted": 0
        }


class CommunicationProtocol:
    """Base class for communication protocols."""
    
    def __init__(self, vocab_size: int = 10, message_length: int = 5):
        """Initialize protocol.
        
        Args:
            vocab_size: Size of communication vocabulary
            message_length: Maximum length of encoded messages
        """
        self.vocab_size = vocab_size
        self.message_length = message_length
    
    def encode_proposal(self, proposal_data: Dict[str, Any]) -> Message:
        """Encode a proposal into a message.
        
        Args:
            proposal_data: Data to encode
            
        Returns:
            Encoded message
        """
        raise NotImplementedError
    
    def decode_response(self, message: Message) -> Any:
        """Decode a response message.
        
        Args:
            message: Message to decode
            
        Returns:
            Decoded response
        """
        raise NotImplementedError


class NegotiationProtocol(CommunicationProtocol):
    """Protocol for resource allocation negotiation."""
    
    def __init__(self, vocab_size: int = 10, message_length: int = 5):
        super().__init__(vocab_size, message_length)
        
    def encode_proposal(self, proposal_data: Dict[str, Any]) -> Message:
        """Encode a resource allocation proposal."""
        # Discretize resource split
        if "resource_split" in proposal_data:
            split = proposal_data["resource_split"]
            discretized = self._discretize_values(split)
        else:
            discretized = [0] * self.message_length
        
        content = {
            "proposal_type": "resource_allocation",
            "encoded_data": discretized,
            "raw_data": proposal_data
        }
        
        return Message(
            sender_id=proposal_data.get("sender_id", 0),
            receiver_id=proposal_data.get("receiver_id"),
            message_type=MessageType.PROPOSAL,
            content=content
        )
    
    def decode_response(self, message: Message) -> Any:
        """Decode a negotiation response."""
        if message.message_type == MessageType.ACCEPT:
            return {"response": "accept", "data": message.content}
        elif message.message_type == MessageType.REJECT:
            return {"response": "reject", "data": message.content}
        elif message.message_type == MessageType.COUNTER:
            counter_offer = self._parse_counter_offer(message.content)
            return {"response": "counter", "data": counter_offer}
        else:
            return {"response": "unknown", "data": message.content}
    
    def _discretize_values(self, values: List[float]) -> List[int]:
        """Discretize continuous values to vocab indices."""
        discretized = []
        
        for value in values:
            # Map to vocab range
            normalized = max(0, min(1, value))  # Clamp to [0, 1]
            discrete_value = int(normalized * (self.vocab_size - 1))
            discretized.append(discrete_value)
        
        # Pad or truncate to message length
        while len(discretized) < self.message_length:
            discretized.append(0)
        
        return discretized[:self.message_length]
    
    def _parse_counter_offer(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a counter-offer from message content."""
        if "encoded_data" in content:
            # Convert discrete values back to continuous
            encoded = content["encoded_data"]
            continuous = [val / (self.vocab_size - 1) for val in encoded]
            return {"resource_split": continuous}
        
        return content.get("raw_data", {})


class CoordinationProtocol(CommunicationProtocol):
    """Protocol for task coordination and assignment."""
    
    def __init__(self, vocab_size: int = 20, message_length: int = 8):
        super().__init__(vocab_size, message_length)
        
        # Define coordination vocabulary
        self.task_types = ["explore", "guard", "collect", "build", "scout"]
        self.priorities = ["low", "medium", "high", "urgent"]
        self.locations = ["north", "south", "east", "west", "center"]
    
    def encode_task_assignment(self, task_data: Dict[str, Any]) -> Message:
        """Encode a task assignment message."""
        encoded_task = self._encode_task_info(task_data)
        
        content = {
            "message_type": "task_assignment",
            "encoded_task": encoded_task,
            "task_data": task_data
        }
        
        return Message(
            sender_id=task_data.get("sender_id", 0),
            receiver_id=task_data.get("receiver_id"),
            message_type=MessageType.REQUEST,
            content=content,
            priority=self._get_priority_level(task_data.get("priority", "medium"))
        )
    
    def encode_status_update(self, status_data: Dict[str, Any]) -> Message:
        """Encode a status update message."""
        content = {
            "message_type": "status_update",
            "status": status_data.get("status", "working"),
            "progress": status_data.get("progress", 0.0),
            "location": status_data.get("location", [0, 0]),
            "needs_help": status_data.get("needs_help", False)
        }
        
        return Message(
            sender_id=status_data.get("sender_id", 0),
            receiver_id=status_data.get("receiver_id"),
            message_type=MessageType.INFO,
            content=content
        )
    
    def decode_response(self, message: Message) -> Any:
        """Decode coordination response."""
        if message.message_type == MessageType.ACCEPT:
            return {
                "response": "task_accepted",
                "agent_id": message.sender_id,
                "data": message.content
            }
        elif message.message_type == MessageType.INFO:
            return {
                "response": "status_update",
                "agent_id": message.sender_id,
                "data": message.content
            }
        elif message.message_type == MessageType.REQUEST:
            return {
                "response": "help_request",
                "agent_id": message.sender_id,
                "data": message.content
            }
        else:
            return {
                "response": "unknown",
                "agent_id": message.sender_id,
                "data": message.content
            }
    
    def _encode_task_info(self, task_data: Dict[str, Any]) -> List[int]:
        """Encode task information into discrete values."""
        encoded = []
        
        # Encode task type
        task_type = task_data.get("task_type", "explore")
        task_idx = self.task_types.index(task_type) if task_type in self.task_types else 0
        encoded.append(task_idx)
        
        # Encode priority
        priority = task_data.get("priority", "medium")
        priority_idx = self.priorities.index(priority) if priority in self.priorities else 1
        encoded.append(priority_idx)
        
        # Encode location (discretized)
        location = task_data.get("location", [0, 0])
        location_encoded = [
            min(self.vocab_size - 1, max(0, int(location[0] / 100))),
            min(self.vocab_size - 1, max(0, int(location[1] / 100)))
        ]
        encoded.extend(location_encoded)
        
        # Pad to message length
        while len(encoded) < self.message_length:
            encoded.append(0)
        
        return encoded[:self.message_length]
    
    def _get_priority_level(self, priority_str: str) -> int:
        """Convert priority string to numeric level."""
        priority_map = {"low": 1, "medium": 2, "high": 3, "urgent": 5}
        return priority_map.get(priority_str, 2)


class EmergencyProtocol(CommunicationProtocol):
    """Protocol for emergency communications."""
    
    def __init__(self):
        super().__init__(vocab_size=5, message_length=3)
        
        self.emergency_types = {
            "obstacle": 0,
            "predator": 1,
            "resource_depletion": 2,
            "system_failure": 3,
            "collision": 4
        }
    
    def encode_emergency(self, emergency_data: Dict[str, Any]) -> Message:
        """Encode an emergency broadcast."""
        emergency_type = emergency_data.get("type", "obstacle")
        type_code = self.emergency_types.get(emergency_type, 0)
        
        location = emergency_data.get("location", [0, 0])
        severity = emergency_data.get("severity", 0.5)
        
        content = {
            "emergency_type": emergency_type,
            "type_code": type_code,
            "location": location,
            "severity": severity,
            "timestamp": time.time(),
            "reporter_id": emergency_data.get("reporter_id", 0)
        }
        
        return Message(
            sender_id=emergency_data.get("sender_id", 0),
            receiver_id=None,  # Broadcast
            message_type=MessageType.EMERGENCY,
            content=content,
            priority=5,  # Maximum priority
            ttl=20  # Longer TTL for emergencies
        )
    
    def decode_response(self, message: Message) -> Any:
        """Decode emergency response."""
        return {
            "response": "emergency_alert",
            "emergency_data": message.content
        }