# Consiva chat on uat.consiva.ai: switch to the hosted widget

**For:** the uat.consiva.ai developer
**Effort:** about 15 minutes, one time only
**Result:** UAT shows the same Consiva chat as https://rag.consiva.tech. From then on, every chat change made by the Consiva team appears on UAT automatically. You never need to touch chat code again.

---

## Why this change

Today UAT has its own copy of the chat widget. The API (`https://apirag.consiva.tech`) sends the data, but
your widget decides what appears on screen, so every new chat feature (for example the new "next step"
card with a button) needed code changes on your side.

With this change, UAT loads the chat from Consiva's server instead:

```html
<script src="https://rag.consiva.tech/widget.js"></script>
```

That one script draws the whole chat. When Consiva deploys a change, UAT gets it on the next page load.

## What you get

- Streaming answers, sources, and the **next-step card** (for example *"Do you want a free DPDP compliance
  assessment?"* with a **Start free assessment** button).
- The chat runs in an **isolated area (Shadow DOM)**: your site's CSS cannot change the chat, and the
  chat's CSS cannot change your site.
- About 96 KB (compressed), loaded after your page is interactive, so it does not slow down your page.

> **Note:** UAT will show the Consiva chat design (the same as rag.consiva.tech), not the current indigo
> widget. The suggestion buttons, copy button and "AI can be wrong" note of your current widget go away
> with it. If they are needed, the Consiva team adds them to the shared widget and both sites get them.

---

## Before you start

Make sure the widget is live. Open this in a browser or run:

```bash
curl -I https://rag.consiva.tech/widget.js
```

You should see `Content-Type: application/javascript` and `Cache-Control: no-cache`.
If you see `text/html`, the Consiva team has not deployed yet: wait for their go-ahead.

---

## Step 1: Remove the current chat widget

Delete the existing chat component (the floating **"Consiva AI Assistant"** with CSS classes starting
with `cb-`) and the place where it is rendered in the layout. Otherwise the page shows two chat buttons.

The `NEXT_PUBLIC_CHATBOT_API_URL` environment variable is no longer needed by the site. Leaving it set does
no harm.

## Step 2: Load the hosted widget once, in the root layout

**Next.js (App Router), `app/layout.tsx`:**

```tsx
import Script from "next/script"

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
        <Script src="https://rag.consiva.tech/widget.js" strategy="afterInteractive" />
      </body>
    </html>
  )
}
```

**Next.js (Pages Router):** add the same `<Script ... strategy="afterInteractive" />` in `pages/_app.tsx`,
next to `<Component {...pageProps} />`.

**Plain HTML:** `<script src="https://rag.consiva.tech/widget.js" defer></script>` before `</body>`.

Add it only once (in the layout, not per page). If it is loaded twice by mistake, the widget still shows
only one chat.

## Step 3: Allow it in the Content-Security-Policy

UAT sets its CSP in **two places**, and a script must be allowed by **both**:

1. the `Content-Security-Policy` **response header** (in `next.config` `headers()`, `middleware`, or the
   hosting / proxy config), and
2. the `<meta http-equiv="Content-Security-Policy">` **tag** in the page.

| Where | Directive | Change |
|---|---|---|
| Response header | `script-src` | **add** `https://rag.consiva.tech` |
| Meta tag | `script-src` | **add** `https://rag.consiva.tech` (keep the existing `sha256-…` hashes) |
| Response header | `connect-src` | **replace** `https://apirag.consiva.tech/ask/stream` **with** `https://apirag.consiva.tech` |

Example, header `connect-src`:

```diff
- connect-src 'self' https://api.consiva.ai https://uatapi.consiva.ai https://apirag.consiva.tech/ask/stream ...
+ connect-src 'self' https://api.consiva.ai https://uatapi.consiva.ai https://apirag.consiva.tech ...
```

`https://apirag.consiva.tech` (no path) allows every chat endpoint, including the streaming one and its
fallback. The meta tag has no `connect-src`, so nothing to change there.

Nothing else is needed: the Consiva API already accepts requests from `https://uat.consiva.ai` (CORS).

---

## Check after deploying

1. One chat button appears, bottom right.
2. Ask **"how consiva help in dpdp compliance"**: the answer streams in as bullet points, sources appear,
   then a card **"Do you want a free DPDP compliance assessment?"** with a **Start free assessment** button
   that opens the free sign-up page.
3. Ask **"I want to book a demo"**: the card's button opens `consiva.ai/contact`.
4. **DevTools → Console**: no `Content-Security-Policy` errors mentioning `rag.consiva.tech` or
   `apirag.consiva.tech`.

## Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| No chat button at all | `script-src` blocks the widget (Console shows a CSP error for `widget.js`) | Add `https://rag.consiva.tech` to `script-src` in **both** the header and the meta tag |
| Chat opens, but answers fail with "could not connect" | `connect-src` blocks the API | Use `https://apirag.consiva.tech` (no path) in `connect-src` |
| Two chat buttons | The old widget is still rendered | Remove the old component (Step 1) |
| `widget.js` returns an HTML page | The widget is not deployed yet | Ask the Consiva team; see "Before you start" |

## How to undo

Remove the `<Script src="https://rag.consiva.tech/widget.js" />` line and restore your old chat component
from version control. The CSP entries can stay; they do no harm.

## After this

Nothing. The Consiva team updates the chat on their side; UAT picks it up automatically. Keep the script
tag and the two CSP entries in place.

---

## Prompt for a coding agent

Paste this into Claude Code, Cursor or Copilot inside the UAT website repository, then review the diff and
run the checks above.

```text
In this Next.js site (uat.consiva.ai), replace the built-in Consiva chat widget with the hosted Consiva widget. Change nothing else.

1. Remove the existing chat widget component (the floating "Consiva AI Assistant", CSS classes starting with "cb-") and every place it is rendered. Delete its now-unused code and styles. Do not remove anything unrelated to the chat.

2. Load the hosted widget once on every page, in the root layout:
   - App Router (app/layout.tsx): import Script from "next/script" and add
     <Script src="https://rag.consiva.tech/widget.js" strategy="afterInteractive" />
     inside <body>, after {children}.
   - Pages Router: add the same <Script> in pages/_app.tsx next to <Component {...pageProps} />.

3. Content-Security-Policy. It is set in two places; update both:
   - The CSP response header (next.config.js/ts headers(), middleware, or hosting/proxy config):
     a) add https://rag.consiva.tech to script-src;
     b) in connect-src, replace https://apirag.consiva.tech/ask/stream with https://apirag.consiva.tech.
   - The <meta http-equiv="Content-Security-Policy"> tag: add https://rag.consiva.tech to script-src and keep all existing sha256 hashes. Do not add connect-src there.

Acceptance checks:
- Exactly one chat button appears, bottom right.
- Asking "how consiva help in dpdp compliance" streams an answer and then shows a "Start free assessment" button that opens the free sign-up page.
- The browser console shows no CSP errors for rag.consiva.tech or apirag.consiva.tech.

When done, list every file you changed and what you changed in it.
```
