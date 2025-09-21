"""
Real web scraping example for FrankensteinDB

This example demonstrates how to integrate FrankensteinDB with actual web scraping
using requests and BeautifulSoup to analyze real websites.
"""

import asyncio
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

# Add the src directory to the path so we can import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import FrankensteinDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper that integrates with FrankensteinDB"""
    
    def __init__(self, db: FrankensteinDB):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_website(self, url: str) -> dict:
        """Scrape a website and extract information"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''
            
            # Extract headings
            headings = []
            for level in range(1, 7):
                for heading in soup.find_all(f'h{level}'):
                    headings.append({
                        'level': level,
                        'text': heading.get_text().strip()
                    })
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                links.append({
                    'url': absolute_url,
                    'text': link.get_text().strip(),
                    'internal': urlparse(absolute_url).netloc == urlparse(url).netloc
                })
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                src = img['src']
                absolute_url = urljoin(url, src)
                images.append({
                    'url': absolute_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
            
            # Detect frameworks and libraries
            frameworks = self._detect_frameworks(soup, response.text)
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'headings': headings,
                'links': links,
                'images': images,
                'frameworks': frameworks,
                'content_length': len(response.content),
                'status_code': response.status_code,
                'scrape_time': time.time()
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def _detect_frameworks(self, soup: BeautifulSoup, html_content: str) -> list:
        """Detect web frameworks and libraries"""
        frameworks = []
        
        # Check for common JavaScript frameworks
        scripts = soup.find_all('script', src=True)
        for script in scripts:
            src = script['src'].lower()
            if 'react' in src:
                frameworks.append('React')
            elif 'vue' in src:
                frameworks.append('Vue.js')
            elif 'angular' in src:
                frameworks.append('Angular')
            elif 'jquery' in src:
                frameworks.append('jQuery')
            elif 'bootstrap' in src:
                frameworks.append('Bootstrap')
        
        # Check for CSS frameworks
        links = soup.find_all('link', href=True)
        for link in links:
            href = link['href'].lower()
            if 'bootstrap' in href:
                frameworks.append('Bootstrap')
            elif 'tailwind' in href:
                frameworks.append('Tailwind CSS')
            elif 'bulma' in href:
                frameworks.append('Bulma')
        
        # Check for meta tags indicating frameworks
        meta_generator = soup.find('meta', attrs={'name': 'generator'})
        if meta_generator:
            generator = meta_generator.get('content', '').lower()
            if 'wordpress' in generator:
                frameworks.append('WordPress')
            elif 'drupal' in generator:
                frameworks.append('Drupal')
            elif 'jekyll' in generator:
                frameworks.append('Jekyll')
        
        return list(set(frameworks))  # Remove duplicates
    
    async def scrape_and_store(self, url: str, content: str = None, user_context: dict = None):
        """Scrape a website and store it in FrankensteinDB"""
        logger.info(f"Scraping and storing: {url}")
        
        # Use provided content or scrape it
        if content is None:
            scraped_data = self.scrape_website(url)
            if not scraped_data:
                return False
            content = f"{scraped_data['title']} - {scraped_data['description']}"
        
        # Store in FrankensteinDB
        result = await self.db.store_website_snapshot(
            url=url,
            content=content,
            user_context=user_context or {}
        )
        
        logger.info(f"Stored snapshot for {url}: {result}")
        return result


async def demo_web_scraping():
    """Demonstrate web scraping with FrankensteinDB"""
    
    # Initialize FrankensteinDB
    db = FrankensteinDB()
    await db.initialize()
    
    # Initialize scraper
    scraper = WebScraper(db)
    
    # List of websites to scrape (using safe, public sites)
    websites = [
        "https://httpbin.org/",
        "https://jsonplaceholder.typicode.com/",
        "https://httpstat.us/",
    ]
    
    logger.info("Starting web scraping demo...")
    
    # Scrape and store websites
    for url in websites:
        try:
            await scraper.scrape_and_store(
                url=url,
                user_context={"scraping_session": "demo", "timestamp": time.time()}
            )
            await asyncio.sleep(1)  # Be respectful with requests
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
    
    # Demonstrate search functionality
    logger.info("\nSearching for websites...")
    search_results = await db.search_websites("json api")
    for result in search_results[:3]:
        logger.info(f"Found: {result['url']} (Score: {result['score']:.2f})")
    
    # Demonstrate domain intelligence
    logger.info("\nGetting domain intelligence...")
    domain_intel = await db.get_domain_intelligence("httpbin.org")
    logger.info(f"Domain intelligence: {domain_intel}")
    
    # Get analytics
    logger.info("\nGetting analytics...")
    analytics = await db.get_analytics()
    logger.info(f"Total websites: {analytics['total_websites']}")
    logger.info(f"Popular frameworks: {analytics['popular_frameworks'][:5]}")
    
    await db.cleanup()
    logger.info("Demo completed!")


async def analyze_website_evolution():
    """Demonstrate tracking website changes over time"""
    
    db = FrankensteinDB()
    await db.initialize()
    
    scraper = WebScraper(db)
    
    # Track a website over time (simulate with different content)
    url = "https://httpbin.org/"
    
    logger.info("Simulating website evolution tracking...")
    
    # Store initial version
    await scraper.scrape_and_store(
        url=url,
        content="Initial version of the API testing service",
        user_context={"version": "1.0"}
    )
    
    await asyncio.sleep(1)
    
    # Store updated version
    await scraper.scrape_and_store(
        url=url,
        content="Updated API testing service with new endpoints",
        user_context={"version": "1.1"}
    )
    
    await asyncio.sleep(1)
    
    # Store another version
    await scraper.scrape_and_store(
        url=url,
        content="Enhanced API testing service with improved documentation",
        user_context={"version": "1.2"}
    )
    
    # Get evolution timeline
    logger.info("\nWebsite evolution timeline:")
    timeline = await db.scylla_emulator.get_evolution_timeline(url)
    for entry in timeline:
        logger.info(f"Version {entry.get('metadata', {}).get('version', 'unknown')}: "
                   f"{entry['timestamp']} - Size: {entry['size']} chars")
    
    await db.cleanup()


if __name__ == "__main__":
    print("FrankensteinDB Web Scraping Examples")
    print("====================================")
    
    print("\n1. Basic web scraping demo")
    asyncio.run(demo_web_scraping())
    
    print("\n2. Website evolution tracking")
    asyncio.run(analyze_website_evolution())
from typing import Dict, List
import re

# Optional imports - install with: pip install requests beautifulsoup4
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("⚠️  Web scraping dependencies not available")
    print("   Install with: pip install requests beautifulsoup4")

from src import FrankensteinDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebsiteAnalyzer:
    """Advanced website analyzer with real web scraping capabilities"""
    
    def __init__(self, frankenstein_db: FrankensteinDB):
        self.db = frankenstein_db
        self.session = requests.Session() if SCRAPING_AVAILABLE else None
        if self.session:
            self.session.headers.update({
                'User-Agent': 'FrankensteinDB/1.0 (Website Analysis Bot)'
            })
    
    async def scrape_and_analyze(self, url: str, user_id: str = None) -> Dict:
        """
        Scrape a website and perform comprehensive analysis
        
        Args:
            url: Website URL to scrape
            user_id: Optional user ID for context tracking
            
        Returns:
            Analysis results dictionary
        """
        if not SCRAPING_AVAILABLE:
            print(f"❌ Cannot scrape {url} - dependencies not installed")
            return {'error': 'Scraping dependencies not available'}
        
        try:
            logger.info(f"🕷️  Scraping {url}")
            
            # Fetch the webpage
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html_content = response.text
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Analyze the website
            fingerprint = self._analyze_website(soup, html_content, url)
            
            # Extract additional metadata
            metadata = self._extract_metadata(soup, response)
            
            # Store in FrankensteinDB
            dna = await self.db.store_website_snapshot(
                url,
                html_content,
                fingerprint,
                user_context=user_id,
                keywords=metadata.get('keywords', [])
            )
            
            # Compile analysis results
            analysis = {
                'url': url,
                'dna': dna.to_dict(),
                'metadata': metadata,
                'fingerprint': fingerprint,
                'analysis_timestamp': time.time(),
                'status': 'success'
            }
            
            logger.info(f"✅ Analysis complete for {url}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'status': 'error',
                'analysis_timestamp': time.time()
            }
    
    def _analyze_website(self, soup: BeautifulSoup, html_content: str, url: str) -> Dict:
        """Comprehensive website analysis"""
        
        # Element counting
        element_counts = {}
        for element in soup.find_all():
            tag = element.name
            element_counts[tag] = element_counts.get(tag, 0) + 1
        
        # Framework detection
        frameworks = self._detect_frameworks(soup, html_content)
        
        # Page type detection
        page_type = self._detect_page_type(soup, url)
        
        # Accessibility analysis
        accessibility_features = self._analyze_accessibility(soup)
        
        # Performance analysis
        performance_hints = self._analyze_performance(soup, html_content)
        
        # SEO analysis
        seo_data = self._analyze_seo(soup)
        
        return {
            'dom_depth': self._calculate_dom_depth(soup),
            'element_counts': element_counts,
            'framework_signatures': frameworks,
            'page_type': page_type,
            'accessibility_features': accessibility_features,
            'performance_hints': performance_hints,
            'seo_data': seo_data
        }
    
    def _detect_frameworks(self, soup: BeautifulSoup, html_content: str) -> List[str]:
        """Detect web frameworks and libraries"""
        frameworks = []
        
        # React detection
        if (soup.find(attrs={'data-reactroot': True}) or 
            'react' in html_content.lower() or
            'ReactDOM' in html_content):
            frameworks.append('react')
        
        # Vue.js detection
        if ('vue' in html_content.lower() or
            soup.find(attrs={'id': 'app'}) and 'Vue' in html_content):
            frameworks.append('vue')
        
        # Angular detection
        if ('angular' in html_content.lower() or
            'ng-app' in html_content or
            '@angular' in html_content):
            frameworks.append('angular')
        
        # jQuery detection
        if ('jquery' in html_content.lower() or
            '$(' in html_content or
            'jQuery' in html_content):
            frameworks.append('jquery')
        
        # Bootstrap detection
        if ('bootstrap' in html_content.lower() or
            soup.find('link', href=re.compile(r'bootstrap', re.I))):
            frameworks.append('bootstrap')
        
        # Tailwind CSS detection
        if ('tailwind' in html_content.lower() or
            'tailwindcss' in html_content):
            frameworks.append('tailwind')
        
        # Webpack detection
        if ('webpack' in html_content.lower() or
            '/webpack/' in html_content):
            frameworks.append('webpack')
        
        # Next.js detection
        if ('next' in html_content.lower() and 'js' in html_content.lower()) or '_next' in html_content:
            frameworks.append('nextjs')
        
        # Gatsby detection
        if 'gatsby' in html_content.lower() or 'gatsby-' in html_content:
            frameworks.append('gatsby')
        
        return frameworks
    
    def _detect_page_type(self, soup: BeautifulSoup, url: str) -> str:
        """Detect the type of webpage"""
        title = soup.find('title')
        title_text = title.get_text().lower() if title else ''
        
        # Check URL patterns
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in ['/blog/', '/article/', '/post/']):
            return 'blog'
        elif any(pattern in url_lower for pattern in ['/product/', '/shop/', '/store/']):
            return 'product'
        elif any(pattern in url_lower for pattern in ['/about', '/company']):
            return 'about'
        elif any(pattern in url_lower for pattern in ['/contact', '/support']):
            return 'contact'
        
        # Check content patterns
        body_text = soup.get_text().lower()
        if any(word in title_text for word in ['home', 'welcome']) or url_lower.endswith('/'):
            return 'homepage'
        elif any(word in body_text[:500] for word in ['article', 'blog', 'posted', 'author']):
            return 'blog'
        elif any(word in body_text[:500] for word in ['product', 'price', 'buy', 'purchase']):
            return 'product'
        elif any(word in body_text[:500] for word in ['about', 'company', 'team', 'mission']):
            return 'about'
        elif any(word in body_text[:500] for word in ['contact', 'email', 'phone', 'address']):
            return 'contact'
        
        return 'unknown'
    
    def _analyze_accessibility(self, soup: BeautifulSoup) -> List[str]:
        """Analyze accessibility features"""
        features = []
        
        # Alt text on images
        images = soup.find_all('img')
        if images and all(img.get('alt') for img in images):
            features.append('alt_text')
        
        # ARIA labels
        if soup.find(attrs={'aria-label': True}) or soup.find(attrs={'aria-labelledby': True}):
            features.append('aria_labels')
        
        # Semantic HTML
        semantic_tags = ['nav', 'main', 'header', 'footer', 'article', 'section', 'aside']
        if any(soup.find(tag) for tag in semantic_tags):
            features.append('semantic_html')
        
        # Role attributes
        if soup.find(attrs={'role': True}):
            features.append('roles')
        
        # Tab navigation
        if soup.find(attrs={'tabindex': True}):
            features.append('tab_navigation')
        
        # Form labels
        forms = soup.find_all('form')
        if forms:
            for form in forms:
                inputs = form.find_all(['input', 'textarea', 'select'])
                labels = form.find_all('label')
                if len(labels) >= len(inputs) * 0.8:  # 80% of inputs have labels
                    features.append('form_labels')
                    break
        
        return features
    
    def _analyze_performance(self, soup: BeautifulSoup, html_content: str) -> Dict:
        """Analyze performance characteristics"""
        
        # Count resources
        scripts = soup.find_all('script')
        stylesheets = soup.find_all('link', rel='stylesheet')
        images = soup.find_all('img')
        
        # Check for lazy loading
        lazy_images = [img for img in images if img.get('loading') == 'lazy']
        
        # Check for async/defer scripts
        async_scripts = [script for script in scripts if script.get('async') or script.get('defer')]
        
        # Check for CDN usage
        cdn_patterns = ['cdn.', 'cloudflare', 'amazonaws', 'gstatic', 'jsdelivr', 'unpkg']
        cdn_resources = []
        for script in scripts:
            src = script.get('src', '')
            if any(pattern in src for pattern in cdn_patterns):
                cdn_resources.append(src)
        
        return {
            'total_scripts': len(scripts),
            'external_scripts': len([s for s in scripts if s.get('src')]),
            'async_scripts': len(async_scripts),
            'total_stylesheets': len(stylesheets),
            'total_images': len(images),
            'lazy_images': len(lazy_images),
            'has_lazy_loading': len(lazy_images) > 0,
            'cdn_resources': len(cdn_resources),
            'html_size_kb': len(html_content) / 1024,
            'inline_scripts_count': len([s for s in scripts if not s.get('src')])
        }
    
    def _analyze_seo(self, soup: BeautifulSoup) -> Dict:
        """Analyze SEO characteristics"""
        
        # Title
        title = soup.find('title')
        title_text = title.get_text() if title else ''
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content') if meta_desc else ''
        
        # Headings structure
        headings = {}
        for i in range(1, 7):
            headings[f'h{i}'] = len(soup.find_all(f'h{i}'))
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        meta_data = {}
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                meta_data[name] = content
        
        return {
            'title': title_text,
            'title_length': len(title_text),
            'meta_description': description,
            'description_length': len(description),
            'headings_structure': headings,
            'meta_tags_count': len(meta_data),
            'has_og_tags': any(key.startswith('og:') for key in meta_data.keys()),
            'has_twitter_tags': any(key.startswith('twitter:') for key in meta_data.keys())
        }
    
    def _calculate_dom_depth(self, soup: BeautifulSoup) -> int:
        """Calculate maximum DOM depth"""
        def get_depth(element, current_depth=0):
            if not element.children:
                return current_depth
            return max(get_depth(child, current_depth + 1) 
                      for child in element.children 
                      if hasattr(child, 'children'))
        
        try:
            return get_depth(soup.body) if soup.body else get_depth(soup)
        except:
            return 0
    
    def _extract_metadata(self, soup: BeautifulSoup, response) -> Dict:
        """Extract additional metadata"""
        
        # Extract keywords from content
        text = soup.get_text().lower()
        words = re.findall(r'\b\w{4,}\b', text)  # Words with 4+ characters
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top keywords (excluding common words)
        common_words = {'that', 'this', 'with', 'have', 'will', 'your', 'they', 'been', 'their', 'said', 'each', 'which', 'them', 'than', 'many', 'some', 'like', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'life', 'only', 'new', 'years', 'way', 'may', 'say', 'come', 'its', 'now', 'find', 'any', 'these', 'give', 'day', 'most', 'us'}
        
        keywords = [word for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20] 
                   if word not in common_words and len(word) > 4]
        
        return {
            'keywords': keywords[:10],  # Top 10 keywords
            'response_size': len(response.content),
            'content_type': response.headers.get('content-type', ''),
            'server': response.headers.get('server', ''),
            'response_time': response.elapsed.total_seconds() if hasattr(response, 'elapsed') else None,
            'status_code': response.status_code,
            'word_count': len(words),
            'character_count': len(text)
        }


async def main():
    """Advanced web scraping and analysis example"""
    
    if not SCRAPING_AVAILABLE:
        print("❌ Cannot run advanced example - web scraping dependencies not installed")
        print("   Install with: pip install requests beautifulsoup4")
        return
    
    print("🧟‍♂️ Initializing Advanced Website Intelligence System...")
    
    # Initialize FrankensteinDB
    frankenstein = FrankensteinDB()
    
    # Initialize analyzer
    analyzer = WebsiteAnalyzer(frankenstein)
    
    # Example websites to scrape and analyze
    websites = [
        'https://httpbin.org/html',  # Test site that returns HTML
        'https://example.com',        # Simple example site
        # Add more URLs here for real analysis
    ]
    
    print(f"\n🕷️  Starting analysis of {len(websites)} websites...")
    
    # Analyze each website
    results = []
    for i, url in enumerate(websites):
        print(f"\n📊 Analyzing website {i+1}/{len(websites)}: {url}")
        
        result = await analyzer.scrape_and_analyze(url, user_id=f"analyst_{i}")
        results.append(result)
        
        if result['status'] == 'success':
            dna = result['dna']
            metadata = result['metadata']
            fingerprint = result['fingerprint']
            
            print(f"   ✅ Analysis successful")
            print(f"   📦 DNA compressed to {dna['compressed_size']} bytes")
            print(f"   🏗️  Frameworks: {fingerprint['framework_signatures']}")
            print(f"   📄 Page type: {fingerprint['page_type']}")
            print(f"   ♿ Accessibility features: {len(fingerprint['accessibility_features'])}")
            print(f"   ⚡ Performance score: {len(fingerprint['performance_hints'])} metrics")
            print(f"   🔍 Keywords: {metadata['keywords'][:5]}")
            
        else:
            print(f"   ❌ Analysis failed: {result['error']}")
    
    print(f"\n📈 Analysis complete! Processed {len([r for r in results if r['status'] == 'success'])}/{len(results)} websites successfully")
    
    # Demonstrate intelligence capabilities
    if any(r['status'] == 'success' for r in results):
        print("\n🧠 Demonstrating intelligence capabilities...")
        
        # Get domain intelligence for first successful site
        successful_result = next(r for r in results if r['status'] == 'success')
        domain = successful_result['url'].replace('https://', '').replace('http://', '').split('/')[0]
        
        intelligence = await frankenstein.get_domain_intelligence(domain)
        print(f"   🎯 Domain {domain} intelligence:")
        print(f"      - Statistics: {intelligence['statistics']}")
        print(f"      - Evolution: {len(intelligence['evolution_timeline'])} snapshots")
        
        # Search functionality
        search_results = await frankenstein.search_websites('example')
        print(f"   🔍 Search for 'example': {len(search_results)} results")
        
        # Framework trends
        if any('react' in r.get('fingerprint', {}).get('framework_signatures', []) for r in results if r['status'] == 'success'):
            react_trends = await frankenstein.get_framework_trends('react')
            print(f"   📊 React trends: {react_trends['total_sites_using']} sites using React")
    
    print("\n🏥 System health check...")
    health = await frankenstein.get_system_health()
    print(f"   Components: {health['components']}")
    
    print("\n🧹 Cleaning up...")
    await frankenstein.close()
    
    print("\n🎉 Advanced analysis complete!")
    print("   The system is ready for production web intelligence gathering!")


if __name__ == "__main__":
    asyncio.run(main())