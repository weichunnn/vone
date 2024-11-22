const TurndownService = require('turndown');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const cheerio = require('cheerio');

async function scrapeUrls(urls) {
  const turndownService = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
  });

  // Create output directory
  const outputDir = 'scraped_content';
  await fs.mkdir(outputDir, { recursive: true });

  for (const url of urls) {
    try {
      console.log(`Scraping: ${url}`);

      // Fetch page content
      const response = await axios.get(url, {
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
        },
      });

      // Parse HTML
      const $ = cheerio.load(response.data);

      // Remove unwanted elements
      $('script, style, nav, footer, header, .sidebar, .ads').remove();

      // Get main content
      const content = $('main, article, .content, .main-content, body')
        .first()
        .html();

      // Convert to markdown
      const markdown = turndownService.turndown(content);

      // Generate filename from URL
      const filename =
        url
          .split('/')
          .pop()
          .replace(/[^a-z0-9]/gi, '_')
          .toLowerCase() + '.md';
      const filepath = path.join(outputDir, filename);

      // Save to file
      await fs.writeFile(filepath, markdown);
      console.log(`Saved: ${filename}`);
    } catch (error) {
      console.error(`Error scraping ${url}:`, error.message);
    }
  }
}

// Example usage with array of URLs
const urls = [
  'https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html',
  'https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html',
  'https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html',
  'https://docs.rapids.ai/api/cudf/stable/user_guide/10min/',
  'https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html',
  'https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html',
];

scrapeUrls(urls).catch(console.error);
