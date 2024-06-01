
from pytz import timezone
from datetime import datetime

KST = timezone('Asia/Seoul')
datetime_now = datetime.now()
localtime_now = datetime_now.astimezone(KST)

print(f"current time = {localtime_now}")

