"""
diagnose_nitro_embeds.py

Diagnostic tool to verify NITRO channel embed extraction
Tests if discord_interface is properly extracting embed content

Run this to see exactly what the bot sees from NITRO messages
"""

import asyncio
import os
from dotenv import load_dotenv
from aiohttp import ClientSession

# Load environment
load_dotenv()

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_AUTH_TOKEN')
NITRO_CHANNEL_ID = "YOUR_NITRO_CHANNEL_ID"  # Update this

async def extract_embed_content(embeds):
    """Same logic as production discord_interface.py"""
    extracted_parts = []
    
    for embed in embeds:
        if embed.get('title'):
            extracted_parts.append(embed['title'])
        
        if embed.get('description'):
            extracted_parts.append(embed['description'])
        
        if embed.get('fields'):
            for field in embed['fields']:
                field_name = field.get('name', '')
                field_value = field.get('value', '')
                
                if field_name and field_value:
                    extracted_parts.append(f"{field_name}: {field_value}")
                elif field_value:
                    extracted_parts.append(field_value)
        
        if embed.get('footer') and embed['footer'].get('text'):
            extracted_parts.append(embed['footer']['text'])
        
        if embed.get('author') and embed['author'].get('name'):
            extracted_parts.append(embed['author']['name'])
    
    combined_text = '\n'.join(extracted_parts)
    combined_text = combined_text.replace('**', '')
    combined_text = combined_text.replace('__', '')
    combined_text = combined_text.replace('```', '')
    
    return combined_text.strip()


async def test_nitro_channel():
    """Test fetching and parsing NITRO messages"""
    
    if not DISCORD_TOKEN:
        print("❌ ERROR: DISCORD_AUTH_TOKEN not found in .env")
        return
    
    if NITRO_CHANNEL_ID == "YOUR_NITRO_CHANNEL_ID":
        print("❌ ERROR: Update NITRO_CHANNEL_ID in script (line 18)")
        return
    
    headers = {
        "Authorization": DISCORD_TOKEN,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    async with ClientSession(headers=headers) as session:
        url = f"https://discord.com/api/v9/channels/{NITRO_CHANNEL_ID}/messages?limit=10"
        
        print("="*80)
        print("🔍 NITRO EMBED DIAGNOSTIC TEST")
        print("="*80)
        print(f"Channel ID: {NITRO_CHANNEL_ID}")
        print("Fetching last 10 messages...\n")
        
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                print(f"❌ ERROR: Status {response.status}")
                print(f"Response: {await response.text()}")
                return
            
            messages = await response.json()
            print(f"✅ Fetched {len(messages)} messages\n")
            
            found_embeds = False
            
            for msg in reversed(messages):
                msg_id = msg['id']
                content = msg.get('content', '')
                embeds = msg.get('embeds', [])
                timestamp = msg.get('timestamp', '')
                
                print("="*80)
                print(f"📨 Message ID: {msg_id}")
                print(f"   Time: {timestamp[:19]}")
                print(f"   Raw Content: '{content[:100]}'")
                
                if embeds:
                    found_embeds = True
                    print(f"\n   🎴 EMBED FOUND ({len(embeds)} embed(s))")
                    
                    # Extract using production logic
                    extracted = await extract_embed_content(embeds)
                    
                    print(f"\n   ╔══ WHAT BOT SEES (msg_content) ══╗")
                    
                    # Combine content and embed like production does
                    if content and extracted:
                        full_content = f"{content}\n{extracted}"
                    elif extracted:
                        full_content = extracted
                    else:
                        full_content = content
                    
                    for line in full_content.split('\n'):
                        print(f"   ║ {line}")
                    print(f"   ╚══════════════════════════════════╝")
                    
                    # Show raw embed structure
                    print(f"\n   📋 RAW EMBED DATA:")
                    for i, embed in enumerate(embeds):
                        print(f"      Embed #{i+1}:")
                        if embed.get('title'):
                            print(f"         title: {embed['title']}")
                        if embed.get('description'):
                            print(f"         description: {embed['description'][:100]}")
                        if embed.get('fields'):
                            print(f"         fields: {len(embed['fields'])} field(s)")
                            for field in embed['fields']:
                                print(f"            {field.get('name')}: {field.get('value')[:50]}")
                        if embed.get('footer'):
                            print(f"         footer: {embed['footer'].get('text')}")
                        if embed.get('author'):
                            print(f"         author: {embed['author'].get('name')}")
                
                else:
                    print("   📝 No embed (text-only message)")
                
                print()
            
            print("="*80)
            print("📊 SUMMARY")
            print("="*80)
            if found_embeds:
                print("✅ EMBED EXTRACTION WORKING")
                print("   If bot still can't parse signals, the issue is in signal_parser.py")
                print("   Check that extracted content contains: ticker, strike, C/P, expiry")
            else:
                print("⚠️  NO EMBEDS FOUND IN LAST 10 MESSAGES")
                print("   Either:")
                print("   1. Last 10 messages don't contain embeds")
                print("   2. Wrong channel ID")
                print("   3. NITRO sends embeds less frequently")
            print("="*80)


if __name__ == "__main__":
    asyncio.run(test_nitro_channel())
