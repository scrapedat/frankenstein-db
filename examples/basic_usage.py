"""
Basic usage example for FrankensteinDB

This example demonstrates the core functionality of the system including:
- Storing website snapshots
- Querying evolution data  
- Searching websites
- Getting domain intelligence
"""

import asyncio
import time
import re
from src import FrankensteinDB


def analyze_html_structure(html_content: str) -> dict:
    """
    Simple HTML analysis to create a website fingerprint
    In a real implementation, this would be much more sophisticated
    """
    # Count different HTML elements
    element_counts = {}
    element_pattern = r'<(\w+)'
    elements = re.findall(element_pattern, html_content.lower())
    for element in elements:
        element_counts[element] = element_counts.get(element, 0) + 1
    
    # Simple framework detection
    frameworks = []
    if 'data-reactroot' in html_content or 'react' in html_content.lower():
        frameworks.append('react')
    if 'vue' in html_content.lower():
        frameworks.append('vue')
    if 'angular' in html_content.lower():
        frameworks.append('angular')
    if 'jquery' in html_content.lower():
        frameworks.append('jquery')
    if 'bootstrap' in html_content.lower():
        frameworks.append('bootstrap')
    if 'webpack' in html_content.lower():
        frameworks.append('webpack')
    
    # Determine page type
    page_type = 'unknown'
    if any(word in html_content.lower() for word in ['home', 'welcome', 'index']):
        page_type = 'homepage'
    elif any(word in html_content.lower() for word in ['about', 'company']):
        page_type = 'about'
    elif any(word in html_content.lower() for word in ['contact', 'email']):
        page_type = 'contact'
    elif any(word in html_content.lower() for word in ['blog', 'article', 'post']):
        page_type = 'blog'
    elif any(word in html_content.lower() for word in ['product', 'shop', 'buy']):
        page_type = 'product'
    
    # Simple accessibility features detection
    accessibility_features = []
    if 'alt=' in html_content:
        accessibility_features.append('alt_text')
    if 'aria-' in html_content:
        accessibility_features.append('aria_labels')
    if any(tag in html_content.lower() for tag in ['<nav', '<main', '<header', '<footer']):
        accessibility_features.append('semantic_html')
    if 'role=' in html_content:
        accessibility_features.append('roles')
    if 'tabindex' in html_content:
        accessibility_features.append('tab_navigation')
    
    # Performance hints
    performance_hints = {
        'has_lazy_loading': 'loading="lazy"' in html_content,
        'inline_scripts_count': len(re.findall(r'<script[^>]*>', html_content)),
        'external_stylesheets': len(re.findall(r'<link[^>]*rel=["\']stylesheet["\']', html_content)),
        'image_count': len(re.findall(r'<img', html_content))
    }
    
    return {
        'dom_depth': len(html_content.split('<')) // 10,  # Rough approximation
        'element_counts': element_counts,
        'framework_signatures': frameworks,
        'page_type': page_type,
        'accessibility_features': accessibility_features,
        'performance_hints': performance_hints
    }


async def main():
    """Example usage of FrankensteinDB"""
    print("🧟‍♂️ Initializing FrankensteinDB...")
    
    # Initialize the monster
    frankenstein = FrankensteinDB()
    
    # Example websites to analyze
    websites = [
        {
            'url': 'https://example.com',
            'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Example React App - Welcome</title>
                <meta name="description" content="A modern React-powered website">
                <link rel="stylesheet" href="/styles.css">
            </head>
            <body>
                <header>
                    <nav aria-label="Main navigation">
                        <a href="/">Home</a>
                        <a href="/about">About</a>
                        <a href="/contact">Contact</a>
                    </nav>
                </header>
                <main>
                    <div id="root" data-reactroot="">
                        <article>
                            <h1>Welcome to our React site</h1>
                            <p>This is a React-powered website with great accessibility.</p>
                            <img src="/hero.jpg" alt="Hero image showing our product" loading="lazy">
                        </article>
                    </div>
                </main>
                <footer>
                    <p>&copy; 2025 Example Corp</p>
                </footer>
                <script src="/bundle.js"></script>
            </body>
            </html>
            '''
        },
        {
            'url': 'https://vue-site.com',
            'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Vue.js Portfolio</title>
                <meta name="description" content="Modern Vue.js portfolio website">
            </head>
            <body>
                <div id="app">
                    <nav>
                        <ul>
                            <li><a href="#home">Home</a></li>
                            <li><a href="#portfolio">Portfolio</a></li>
                            <li><a href="#contact">Contact</a></li>
                        </ul>
                    </nav>
                    <main>
                        <section class="hero">
                            <h1>Vue.js Developer</h1>
                            <p>Building beautiful web applications</p>
                        </section>
                    </main>
                </div>
                <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
                <script src="/app.js"></script>
            </body>
            </html>
            '''
        },
        {
            'url': 'https://blog-site.com/article',
            'html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Understanding Modern Web Architecture</title>
                <meta name="description" content="A deep dive into modern web architecture patterns">
            </head>
            <body>
                <header>
                    <h1>Tech Blog</h1>
                    <nav>
                        <a href="/">Blog</a>
                        <a href="/about">About</a>
                    </nav>
                </header>
                <main>
                    <article role="main">
                        <h1>Understanding Modern Web Architecture</h1>
                        <p>This article explores modern web architecture patterns...</p>
                        <p>Modern frameworks like React, Vue, and Angular have revolutionized...</p>
                        <img src="/diagram.png" alt="Architecture diagram showing microservices">
                    </article>
                </main>
            </body>
            </html>
            '''
        }
    ]
    
    print("\n📝 Storing website snapshots...")
    
    # Store all websites
    stored_dnas = []
    for i, site in enumerate(websites):
        fingerprint = analyze_html_structure(site['html'])
        print(f"   Analyzing {site['url']}...")
        print(f"   - Frameworks detected: {fingerprint['framework_signatures']}")
        print(f"   - Page type: {fingerprint['page_type']}")
        print(f"   - Accessibility features: {len(fingerprint['accessibility_features'])}")
        
        dna = await frankenstein.store_website_snapshot(
            site['url'], 
            site['html'], 
            fingerprint,
            user_context=f"user_{i}",
            keywords=['example', 'demo', 'website']
        )
        
        stored_dnas.append(dna)
        print(f"   ✅ Stored DNA: {dna.structure_hash} ({len(dna.compress())} bytes)")
    
    print("\n📈 Querying website evolution...")
    
    # Query evolution for each domain
    for site in websites:
        domain = site['url'].replace('https://', '').replace('http://', '')
        evolution = await frankenstein.query_website_evolution(domain)
        print(f"   {domain}: {len(evolution)} snapshots found")
        
        if evolution:
            latest = evolution[0]  # Most recent
            print(f"     Latest: {latest['page_type']} with frameworks {latest['frameworks']}")
    
    print("\n🔍 Testing search functionality...")
    
    # Test various searches
    search_queries = [
        'react',
        'vue portfolio',
        'blog article',
        'accessibility'
    ]
    
    for query in search_queries:
        results = await frankenstein.search_websites(query, limit=5)
        print(f"   '{query}' → {len(results)} results")
        for result in results:
            print(f"     - {result['domain']} ({result['page_type']}) - {result['frameworks']}")
    
    print("\n🧠 Getting domain intelligence...")
    
    # Get intelligence for first domain
    domain = websites[0]['url'].replace('https://', '').replace('http://', '')
    intelligence = await frankenstein.get_domain_intelligence(domain)
    
    print(f"   Domain: {intelligence['domain']}")
    print(f"   Statistics: {intelligence['statistics']}")
    print(f"   Evolution timeline: {len(intelligence['evolution_timeline'])} snapshots")
    print(f"   Trends: {intelligence['trends']}")
    
    print("\n📊 Framework trend analysis...")
    
    # Analyze React trend
    react_trends = await frankenstein.get_framework_trends('react')
    print(f"   React usage: {react_trends['total_sites_using']} sites")
    print(f"   Recent adoptions: {react_trends['recent_adoptions']}")
    
    print("\n🏥 System health check...")
    
    # Get system health
    health = await frankenstein.get_system_health()
    print(f"   Overall status: {health['components']}")
    print(f"   Top domains: {len(health['statistics']['top_domains'])}")
    
    print("\n🔄 Testing similar website discovery...")
    
    # Find similar websites
    similar = await frankenstein.find_similar_websites(websites[0]['url'])
    print(f"   Found {len(similar)} similar websites")
    for site in similar:
        print(f"     - {site['domain']} (similarity: {site['similarity_hash']})")
    
    print("\n💾 Testing content caching...")
    
    # Test content retrieval
    for site in websites:
        cached_content = await frankenstein.get_cached_content(site['url'])
        if cached_content:
            print(f"   ✅ Retrieved {len(cached_content)} chars from {site['url']}")
        else:
            print(f"   ❌ No cached content for {site['url']}")
    
    print("\n👤 Testing user context...")
    
    # Test user context storage and retrieval
    await frankenstein.store_user_session('demo_user', {
        'preferences': {'framework': 'react'},
        'last_search': 'react websites',
        'session_start': time.time()
    })
    
    user_context = await frankenstein.get_user_context('demo_user')
    if user_context:
        print(f"   ✅ User context retrieved: {user_context['preferences']}")
    
    print("\n🧹 Cleanup...")
    
    # Clean up resources
    await frankenstein.close()
    
    print("\n🎉 Demo completed successfully!")
    print(f"   Processed {len(websites)} websites")
    print(f"   Generated {len(stored_dnas)} DNA fingerprints")
    print(f"   System is ready for production use!")


if __name__ == "__main__":
    asyncio.run(main())