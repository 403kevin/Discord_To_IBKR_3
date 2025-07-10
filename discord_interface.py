# =============================================================================================== #
from datetime import datetime, timezone, timedelta
import requests
import json
import logging

# =============================================================================================== #

BASE_URL = 'https://discord.com/api/v9/channels/{channel_id}/messages?limit={msg_limit}'

# =============================================================================================== #


class DiscordChannelClient:
    def __init__(self, auth_token: str):
        self.auth_token = auth_token

        self.headers = {'authorization': self.auth_token,
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, '
                                      'like Gecko) Chrome/104.0.5112.79 Safari/537.36'}

    def poll_new_messages(self, channel_id: str, limit: int) -> list:
        try:
            return requests.get(BASE_URL.format(channel_id=channel_id, msg_limit=limit),
                                headers=self.headers).json()
        except Exception as _exc:
            logging.error(f'Exception polling new messages from discord. exc: {_exc}')
            return []


# =============================================================================================== #


if __name__ == '__main__':
    pass

# =============================================================================================== #
