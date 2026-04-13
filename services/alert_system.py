import winsound # Hoặc dùng pygame cho âm thanh phức tạp hơn
import time

class AlertSystem:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 2 # Giây giữa các lần cảnh báo

    def trigger(self, level="low"):
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown:
            return

        if level == "danger":
            # Tiếng kêu kéo dài và gắt cho buồn ngủ
            winsound.Beep(1000, 1000) 
        elif level == "distracted":
            # Tiếng kêu ngắn cho xao nhãng (quay đầu quá lâu)
            winsound.Beep(600, 200)
            
        self.last_alert_time = current_time

    def send_notification(self, msg):
        # Có thể tích hợp gửi Telegram Bot tại đây cho chủ xe/quản lý đội xe
        pass