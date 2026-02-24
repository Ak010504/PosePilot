# voice_feedback.py (THREAD-SAFE VERSION)
import logging
import re
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceFeedback")

class VoiceFeedbackManager:
    def __init__(self):
        logger.info("🎤 Initializing voice...")
        
        self.lock = Lock()  # ✅ Prevents overlapping speech
        
        try:
            import pyttsx3
            test = pyttsx3.init()
            test.stop()
            del test
            
            self.enabled = True
            logger.info("✅ Voice ready")
        except Exception as e:
            logger.error(f"❌ Voice init failed: {e}")
            self.enabled = False
    
    def speak_single(self, text):
        """Speak one message - thread-safe"""
        if not self.enabled:
            return
        
        # ✅ Only one speech at a time
        if not self.lock.acquire(blocking=False):
            logger.warning("⚠️ Already speaking, skipping")
            return
        
        try:
            import pyttsx3
            logger.info(f"🗣️ Speaking: {text}")
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            
            logger.info("✅ Done")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.lock.release()
    
    def speak_batch(self, messages, max_count=3):
        """Combine and speak"""
        if not self.enabled or not messages:
            return
        
        logger.info(f"📋 Combining {len(messages)} messages")
        
        combined = self._create_single_message(messages)
        logger.info(f"🎤 Final: {combined}")
        
        self.speak_single(combined)
    
    def _create_single_message(self, messages):
        """Create ONE natural sentence"""
        if len(messages) == 0:
            return "Good alignment"
        
        if len(messages) == 1:
            return self._naturalize(messages[0])
        
        # Combine bilateral
        combined = self._combine_bilateral(messages)
        combined = combined[:3]  # max 3
        
        if len(combined) == 1:
            return self._naturalize(combined[0])
        elif len(combined) == 2:
            return f"{self._naturalize(combined[0])} and {self._naturalize(combined[1]).lower()}"
        else:
            # "X, Y, and Z"
            parts = [self._naturalize(combined[0])]
            for msg in combined[1:-1]:
                parts.append(self._naturalize(msg).lower())
            parts.append(f"and {self._naturalize(combined[-1]).lower()}")
            return ", ".join(parts)
    
    def _combine_bilateral(self, messages):
        """Combine left+right into both"""
        grouped = {}
        
        for msg in messages:
            action = self._extract_action(msg)
            body_part = self._extract_body_part(msg)
            key = f"{action}_{body_part}"
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(msg)
        
        combined = []
        for key, msgs in grouped.items():
            has_left = any('left' in m.lower() for m in msgs)
            has_right = any('right' in m.lower() for m in msgs)
            
            if has_left and has_right:
                action = self._extract_action(msgs[0])
                body_part = self._extract_body_part(msgs[0])
                combined.append(f"{action} both {body_part}s")
            else:
                combined.append(msgs[0])
        
        return combined
    
    def _naturalize(self, message):
        """Add 'your' for naturalness"""
        msg = message.strip()
        
        if not re.search(r'\byour\b', msg, re.IGNORECASE):
            parts = msg.split(' ', 1)
            if len(parts) == 2:
                msg = f"{parts[0]} your {parts[1]}"
        
        return msg
    
    def _extract_action(self, message):
        match = re.match(r'^(\w+)', message)
        return match.group(1).lower() if match else 'adjust'
    
    def _extract_body_part(self, message):
        parts = ['knee', 'elbow', 'hip', 'shoulder', 'arm', 'leg', 'neck']
        msg_lower = message.lower()
        for part in parts:
            if part in msg_lower:
                return part
        return 'position'
    
    def clear_queue(self):
        pass
    
    def shutdown(self):
        pass

voice_manager = VoiceFeedbackManager()
