
```mermaid
graph LR

subgraph interface
	query--->research
	response
end

subgraph loader
	assets
	text
end

subgraph catalogue
	summaries
	keys
end

subgraph tools
	simplify
	tokenize
	pages
	summarize
	tag
end

assets--->text
text--->|paginate|pages

pages--->summarize--->|summarize|summaries
pages--->tag--->|tag|keys
pages--->tokenize--->simplify--->|store|data

research<--->|search|summaries
research--->|retrieve|keys
keys--->|retrieve|data
data--->|retrieve|response
```
