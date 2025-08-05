// Custom JavaScript for Chainlit UI modifications
(function() {
    'use strict';
    
    // Function to replace "Used LangGraph" with "🤔 생각중입니다"
    function replaceLangGraphText() {
        // Find all text nodes that contain "Used LangGraph"
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    if (node.textContent.includes('Used LangGraph')) {
                        return NodeFilter.FILTER_ACCEPT;
                    }
                    return NodeFilter.FILTER_REJECT;
                }
            }
        );
        
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }
        
        // Replace the text in found nodes
        textNodes.forEach(function(textNode) {
            textNode.textContent = textNode.textContent.replace(/Used LangGraph/g, '🤔 생각중입니다');
        });
    }
    
    // Function to add timing information to steps
    function addTimingInfo() {
        // This will be handled by the Python callback
        // Just ensure the UI updates properly
    }
    
    // Run the replacement function
    function runReplacements() {
        replaceLangGraphText();
        addTimingInfo();
    }
    
    // Initial run
    setTimeout(runReplacements, 100);
    
    // Set up MutationObserver to catch dynamically added content
    const observer = new MutationObserver(function(mutations) {
        let shouldRun = false;
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                // Check if any added nodes contain text
                for (let node of mutation.addedNodes) {
                    if (node.nodeType === Node.TEXT_NODE || 
                        (node.nodeType === Node.ELEMENT_NODE && node.textContent)) {
                        shouldRun = true;
                        break;
                    }
                }
            }
        });
        
        if (shouldRun) {
            setTimeout(runReplacements, 50);
        }
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true
    });
    
    // Also run on common events
    document.addEventListener('DOMContentLoaded', runReplacements);
    window.addEventListener('load', runReplacements);
    
    console.log('Custom Chainlit UI modifications loaded');
})();