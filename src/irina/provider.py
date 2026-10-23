"""Remote OpenAI-compatible chat completion provider for ordinary Irina use.

Inputs: API key str, model name str, endpoint URL str, messages list[dict].
Output: ``generate(messages)`` dict with ``text`` str and ``tokens`` list.
Example: ``RemoteProvider(api_key='...').generate([{'role':'user','content':'hi'}])``
returns an assistant message from the configured service. Credentials are read
from IRINA_API_KEY, with optional legacy environment variable fallback.
"""
import json,os,urllib.request
class RemoteProvider:
    def __init__(self,api_key=None,model=None,url='https://api.deepseek.com/chat/completions'):
        self.api_key=api_key or os.getenv('IRINA_API_KEY') or os.getenv('ODYSSEUS_API_KEY')
        self.model=model or os.getenv('IRINA_MODEL') or os.getenv('ODYSSEUS_MODEL','deepseek-chat')
        self.url=url
    def generate(self,messages):
        if not self.api_key:raise RuntimeError('Set IRINA_API_KEY')
        payload=json.dumps({'model':self.model,'messages':messages}).encode()
        request=urllib.request.Request(self.url,payload,{'Authorization':f'Bearer {self.api_key}',
                                                       'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=90) as response:data=json.load(response)
        return {'text':data['choices'][0]['message'].get('content') or '','tokens':[]}
