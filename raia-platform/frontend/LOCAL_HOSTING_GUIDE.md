# 🚀 RAIA Platform - Local Hosting Guide

## ✅ Current Server Running
**Your RAIA Platform is now live at: http://localhost:8080**

## 🎯 Quick Start Options

### Option 1: serve (Currently Running)
```bash
cd /Users/cladmin/Documents/RAIA/raia-platform/frontend/dist
npx serve -s . -l 8080
```
**→ Open: http://localhost:8080**

### Option 2: Python HTTP Server
```bash
cd /Users/cladmin/Documents/RAIA/raia-platform/frontend/dist
python3 -m http.server 8000
```
**→ Open: http://localhost:8000**

### Option 3: PHP Built-in Server
```bash
cd /Users/cladmin/Documents/RAIA/raia-platform/frontend/dist
php -S localhost:7000
```
**→ Open: http://localhost:7000**

### Option 4: Live Server (VS Code Extension)
1. Install "Live Server" extension in VS Code
2. Right-click on `dist/index.html`
3. Select "Open with Live Server"

## 🔧 Browser Issues Solutions

### If UI doesn't show:
1. **Clear Browser Cache**: Press `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
2. **Try Incognito Mode**: Open a private/incognito window
3. **Check Console**: Press `F12` → Console tab for errors
4. **Try Different Port**: Use a different port number
5. **Disable Extensions**: Turn off ad blockers/security extensions

### Chrome-specific fixes:
```bash
# Open Chrome with disabled security (for local testing)
open -a "Google Chrome" --args --disable-web-security --disable-features=VizDisplayCompositor --user-data-dir=/tmp/chrome_dev_test
```

### Firefox-specific:
- Go to `about:config`
- Set `security.fileuri.strict_origin_policy` to `false`

## 🌐 Online Hosting (Free)

### Vercel (Recommended)
1. Go to [vercel.com](https://vercel.com)
2. Drag your `dist` folder to deploy
3. Get instant live URL

### GitHub Pages
1. Create new GitHub repository
2. Upload `dist` contents to `gh-pages` branch
3. Enable Pages in repository settings

### Surge.sh
```bash
npm install -g surge
cd /Users/cladmin/Documents/RAIA/raia-platform/frontend/dist
surge
```

## 🎉 What You'll See
- **Full RAIA Dashboard** with navigation
- **3 Sample Models** with metrics
- **25+ Working Pages** 
- **Interactive Charts & Analytics**
- **No Authentication Required**
- **Complete UI/UX Experience**

---
**Status**: ✅ Ready to View  
**Current URL**: http://localhost:8080  
**Updated**: 2025-07-22