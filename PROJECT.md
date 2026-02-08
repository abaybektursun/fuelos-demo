# FuelOS Demo Booth

**Status:** Exploratory  
**Hardware:** Reachy Mini + MacBook  
**Voice:** 11Labs (latest personalities)

## Concept

A demo booth where Reachy Mini acts as the face of FuelOS. Robot greets people, demos the app on the MacBook screen, answers questions, and keeps things fun.

## The Vibe

Not a sterile product demo. More like a curious robot who genuinely thinks FuelOS is cool and wants to show you why. Self-aware humor. Acknowledges it's a robot. Doesn't take itself too seriously but knows the product inside out.

Think: if WALL-E was a tech evangelist.

## Components

### 1. Reachy Mini (Physical Presence)
- Head tracking — looks at whoever's talking
- Expressive movements — nods, tilts, antenna wiggles
- Reacts to conversation — surprised, thinking, excited
- Idle behaviors when nobody's around

### 2. MacBook Screen (App Display)
- Shows FuelOS running (simulator or screen mirror from iPhone)
- Can highlight features as Reachy talks about them
- Possibly: automated demo sequences triggered by conversation

### 3. Voice (11Labs)
- Pick a personality that's warm, slightly quirky, not corporate
- Fast response times for natural conversation
- Should sound like someone you'd want to hang out with

### 4. Brain (LLM + Context)
- Knows FuelOS deeply — features, benefits, use cases
- Can answer technical questions
- Has personality — opinions, humor, self-awareness
- Handles objections naturally

## Interaction Flow

```
[Person approaches]
    ↓
Reachy notices (camera/motion) → looks at them → greeting
    ↓
[Conversation starts]
    ↓
Listen → understand intent → respond + move expressively
    ↓
If demo request → guide through app on screen
If question → answer naturally
If just chatting → be charming
    ↓
[Person leaves]
    ↓
Wave goodbye → return to idle/attract mode
```

## Attract Mode (Nobody Around)

- Subtle movements — looking around, antenna twitches
- Maybe occasional comments: "Anyone curious about what's on this screen?"
- Not aggressive. More like a friendly presence that's available.

## Technical Architecture

```
                    ┌─────────────────┐
                    │   11Labs TTS    │
                    └────────┬────────┘
                             │ audio
    ┌────────────────────────┼────────────────────────┐
    │                        ▼                        │
    │  ┌─────────┐    ┌───────────┐    ┌──────────┐  │
    │  │  Mics   │───▶│   LLM +   │───▶│  Reachy  │  │
    │  │(Reachy) │    │  Daemon   │    │  Motors  │  │
    │  └─────────┘    └─────┬─────┘    └──────────┘  │
    │                       │                        │
    │                       ▼                        │
    │               ┌──────────────┐                 │
    │               │ MacBook App  │                 │
    │               │  Controller  │                 │
    │               └──────────────┘                 │
    └────────────────────────────────────────────────┘
```

## FuelOS Knowledge Base (TODO)

Need to dump:
- What FuelOS is / does
- Core features
- Target users
- Differentiators
- Common questions
- Pricing/availability

## Open Questions

1. **Voice selection** — Which 11Labs personality? Need to test a few.
2. **Screen control** — How to automate MacBook display? AppleScript? Playwright?
3. **Wake word** — Should Reachy respond to a name? Or just proximity?
4. **Demo script** — Fully scripted or improvised based on conversation?
5. **Fallback** — What happens when Reachy doesn't know something?

## Next Steps

1. [ ] Get Reachy Mini connected and responding
2. [ ] Test 11Labs voices — find the right personality
3. [ ] Document FuelOS knowledge base
4. [ ] Build simple conversation loop (mic → LLM → voice → motors)
5. [ ] Add screen control for demo sequences
6. [ ] Refine personality prompts
7. [ ] Test with real humans

## Files

```
fuelos-demo/
├── PROJECT.md          # This file
├── PERSONALITY.md      # Voice/character definition
├── KNOWLEDGE.md        # FuelOS info dump
├── src/
│   ├── main.py         # Entry point
│   ├── conversation.py # LLM + voice loop
│   ├── robot.py        # Reachy Mini control
│   └── screen.py       # MacBook automation
└── prompts/
    └── system.txt      # LLM system prompt
```
