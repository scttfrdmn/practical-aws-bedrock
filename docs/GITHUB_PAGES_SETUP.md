# GitHub Pages Setup for Practical AWS Bedrock

This document outlines the setup process for the GitHub Pages site that will host the Practical AWS Bedrock content.

## Architecture Overview

The GitHub Pages site will use Jekyll with a customized theme to present the content in an accessible, navigable format. The site will be responsive and organized to highlight practical applications and step-by-step tutorials.

## File Structure

```
docs/
├── _config.yml               # Jekyll configuration
├── _layouts/                 # Custom layouts
│   ├── default.html          # Base layout with navigation
│   ├── chapter.html          # Layout for chapter pages
│   └── tutorial.html         # Layout for tutorials
├── _includes/                # Reusable components
│   ├── header.html           # Site header
│   ├── footer.html           # Site footer
│   ├── navigation.html       # Navigation menu
│   └── toc.html              # Table of contents
├── assets/                   # Static assets
│   ├── css/                  # Custom styles
│   ├── js/                   # JavaScript for interactive elements
│   └── images/               # Images and diagrams
├── _data/                    # Site data
│   ├── navigation.yml        # Navigation structure
│   └── toc.yml               # Table of contents data
├── index.md                  # Home page
├── learning-path.md          # Learning path guide
└── chapters/                 # Content pages
    ├── getting-started/      # Getting started content
    ├── core-methods/         # Core methods content
    ├── advanced-techniques/  # Advanced techniques content
    └── ...                   # Other content sections
```

## Jekyll Configuration

The `_config.yml` file will include:

```yaml
# Site settings
title: Practical AWS Bedrock
description: A comprehensive, action-oriented guide to AWS Bedrock
url: "https://your-username.github.io/practical-aws-bedrock"
repository: your-username/practical-aws-bedrock
theme: jekyll-theme-minimal  # Base theme to customize

# Build settings
markdown: kramdown
highlighter: rouge
plugins:
  - jekyll-feed
  - jekyll-seo-tag
  - jekyll-sitemap
  - jekyll-toc

# Content settings
collections:
  chapters:
    output: true
    permalink: /:collection/:path/
  tutorials:
    output: true
    permalink: /:collection/:path/

# Default front matter
defaults:
  - scope:
      path: ""
    values:
      layout: default
  - scope:
      path: ""
      type: "chapters"
    values:
      layout: chapter
  - scope:
      path: ""
      type: "tutorials"
    values:
      layout: tutorial
```

## Navigation System

The navigation will be problem-oriented, allowing users to find content based on what they're trying to accomplish. The main navigation categories will be:

1. **Getting Started** - For new users
2. **Core Methods** - Basic implementation patterns
3. **Advanced Techniques** - More complex patterns
4. **Optimization** - Performance and cost optimization
5. **Solutions** - Complete application examples

The navigation will be defined in `_data/navigation.yml`:

```yaml
main:
  - title: "Getting Started"
    url: /chapters/getting-started/
    children:
      - title: "Setting Up AWS Bedrock"
        url: /chapters/getting-started/setup/
      - title: "Understanding Foundation Models"
        url: /chapters/getting-started/foundation-models/
  
  - title: "Core Methods"
    url: /chapters/core-methods/
    children:
      - title: "Synchronous Inference"
        url: /chapters/core-methods/synchronous/
      - title: "Streaming Inference"
        url: /chapters/core-methods/streaming/
      # More items...

  # More categories...
```

## Custom Layouts

Custom layouts will ensure a consistent presentation with practical components:

### Chapter Layout (`_layouts/chapter.html`)

```html
---
layout: default
---

<div class="chapter">
  <div class="chapter-header">
    <h1>{{ page.title }}</h1>
    <div class="chapter-metadata">
      <span class="difficulty">{{ page.difficulty }}</span>
      <span class="time-estimate">{{ page.time-estimate }}</span>
    </div>
  </div>
  
  {{ content }}
  
  <div class="chapter-navigation">
    {% if page.previous %}
    <a href="{{ page.previous.url }}" class="prev">← {{ page.previous.title }}</a>
    {% endif %}
    
    {% if page.next %}
    <a href="{{ page.next.url }}" class="next">{{ page.next.title }} →</a>
    {% endif %}
  </div>
  
  <div class="chapter-resources">
    <h3>Additional Resources</h3>
    <ul>
      {% for resource in page.resources %}
      <li><a href="{{ resource.url }}">{{ resource.title }}</a></li>
      {% endfor %}
    </ul>
  </div>
</div>
```

## Interactive Elements

The site will include several interactive elements:

### Code Blocks with Copy Functionality

```javascript
document.addEventListener('DOMContentLoaded', () => {
  const codeBlocks = document.querySelectorAll('pre.highlight');
  
  codeBlocks.forEach(block => {
    const copyButton = document.createElement('button');
    copyButton.className = 'copy-button';
    copyButton.textContent = 'Copy';
    
    copyButton.addEventListener('click', () => {
      const code = block.querySelector('code').textContent;
      navigator.clipboard.writeText(code);
      
      copyButton.textContent = 'Copied!';
      setTimeout(() => {
        copyButton.textContent = 'Copy';
      }, 2000);
    });
    
    block.appendChild(copyButton);
  });
});
```

### Interactive Diagrams

We'll use [Mermaid.js](https://mermaid-js.github.io/mermaid/) for diagrams:

```html
<div class="mermaid">
  graph TD
    A[Client] --> B[AWS Bedrock]
    B -->|Synchronous| C[InvokeModel]
    B -->|Streaming| D[InvokeModelWithResponseStream]
    B -->|Asynchronous| E[CreateModelInvocationJob]
</div>

<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>
```

### Search Functionality

We'll implement a client-side search using [Lunr.js](https://lunrjs.com/):

```javascript
// Generate search index
const searchIndex = lunr(function() {
  this.ref('id');
  this.field('title', { boost: 10 });
  this.field('content');
  
  // Add documents to index
  documents.forEach(doc => {
    this.add(doc);
  });
});

// Search function
function search(query) {
  return searchIndex.search(query).map(result => {
    return documents.find(doc => doc.id === result.ref);
  });
}
```

## Responsive Design

The site will use responsive design principles to ensure good usability on all devices:

```css
/* Base styles */
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  line-height: 1.6;
  margin: 0;
  padding: 0;
  color: #333;
}

/* Responsive layout */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

/* Responsive navigation */
@media (max-width: 768px) {
  .navigation {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    position: static;
  }
  
  .content {
    margin-left: 0;
  }
}

/* Code blocks */
pre {
  overflow-x: auto;
  background: #f5f7f9;
  border-radius: 5px;
  padding: 15px;
}
```

## Implementation Plan

1. **Initial Setup**
   - Create basic Jekyll structure
   - Configure GitHub Pages in repository settings
   - Set up custom domain (if applicable)

2. **Theme Customization**
   - Customize base theme
   - Implement responsive design
   - Create custom layouts

3. **Content Integration**
   - Convert Markdown content to use front matter
   - Organize content into collections
   - Set up navigation structure

4. **Interactive Features**
   - Implement code copying functionality
   - Set up Mermaid.js for diagrams
   - Add search functionality

5. **Testing and Optimization**
   - Test on multiple devices and browsers
   - Optimize performance
   - Verify all links and navigation

## Deployment

The site will be deployed automatically from the `main` branch using GitHub Pages. The process is:

1. Push changes to the `main` branch
2. GitHub Actions builds the Jekyll site
3. Site is deployed to `https://your-username.github.io/practical-aws-bedrock`

## Next Steps

1. Create the base Jekyll structure
2. Set up GitHub repository with GitHub Pages enabled
3. Customize theme and navigation
4. Convert initial content to Jekyll format
5. Test and deploy first version