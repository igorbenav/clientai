# Building a Multi-Provider Translator with ClientAI

Let's build something interesting - a translator that can compare responses from different AI providers like OpenAI, Groq, and Replicate. We'll see how ClientAI makes it easy to work with multiple providers and compare their translations side by side.

## Getting Started

First, we'll need ClientAI installed with support for all providers. Open your terminal and run:

```bash
pip install clientai[all]
```

You'll also need API keys for each provider. Set these up as environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export GROQ_API_KEY=your_groq_api_key
export REPLICATE_API_KEY=your_replicate_api_key
```

## Building the Translator

Let's start by importing what we need and setting up our environment:

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from clientai import ClientAI

# Get our API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY', 'your_replicate_api_key_here')
```

Now let's create a simple class to hold our translation results. We'll use Python's dataclass feature to keep it clean and simple:

```python
@dataclass
class Translation:
    """Represents a translation with metadata."""
    text: str
    provider: str
    confidence: float
    model: str
    time_taken: float
```

This Translation class will store not just the translated text, but also useful information about who did the translation, how confident they were, and how long it took. This will help us compare different providers.

Now for the main translator class. Let's build it piece by piece:

```python
class MultiProviderTranslator:
    def __init__(self):
        """Initialize connections to different AI providers."""
        self.providers = {
            'openai': ClientAI('openai', api_key=OPENAI_API_KEY),
            'groq': ClientAI('groq', api_key=GROQ_API_KEY),
            'replicate': ClientAI('replicate', api_key=REPLICATE_API_KEY),
        }
        
        self.default_models = {
            'openai': 'gpt-4',
            'groq': 'mixtral-8x7b-32768',
            'replicate': 'meta/llama-2-70b-chat',
        }
        
        self.supported_languages = self._get_supported_languages()
```

In our initialization, we're setting up connections to each provider using ClientAI. We also define which models we want to use by default - GPT-4 for OpenAI, Mixtral for Groq, and Llama 2 for Replicate. Each has its own strengths, and we'll be able to compare them directly.

Let's add the language support:

```python
def _get_supported_languages(self) -> Dict[str, str]:
    """Get dictionary of supported languages and their codes."""
    return {
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Italian': 'it',
        'Portuguese': 'pt',
        'Russian': 'ru',
        'Japanese': 'ja',
        'Chinese': 'zh',
        'Korean': 'ko'
    }
```

Next, we need to create clear instructions for our AI models. Here's how we'll format our translation requests:

```python
def _create_translation_prompt(
    self, 
    text: str, 
    source_lang: str, 
    target_lang: str
) -> str:
    """Create a clear prompt for translation."""
    return f"""
    Translate the following text from {source_lang} to {target_lang}.
    Provide only the direct translation without explanations or notes.

    Text: {text}
    """
```

Now for the interesting part - getting translations from each provider. We want to handle errors gracefully and track how long each translation takes:

```python
def _get_provider_translation(
    self,
    provider: str,
    text: str,
    source_lang: str,
    target_lang: str,
    model: Optional[str] = None
) -> Translation:
    """Get translation from a specific provider."""
    start_time = time.time()
    
    model = model or self.default_models[provider]
    prompt = self._create_translation_prompt(text, source_lang, target_lang)
    
    try:
        response = self.providers[provider].chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3  # Lower temperature for more consistent translations
        )
        
        time_taken = time.time() - start_time
        confidence = 0.95 if len(response.strip()) > 0 else 0.0
        
        return Translation(
            text=response.strip(),
            provider=provider,
            confidence=confidence,
            model=model,
            time_taken=time_taken
        )
    except Exception as e:
        print(f"Error with {provider}: {str(e)}")
        return Translation(
            text=f"Error: {str(e)}",
            provider=provider,
            confidence=0.0,
            model=model,
            time_taken=time.time() - start_time
        )
```

This method does the heavy lifting for each provider. We time how long the translation takes, use the provider's API through ClientAI, and package everything up in our Translation class. If anything goes wrong, we handle it gracefully and return an error message instead of crashing.

To make our translator faster, we'll run all providers in parallel:

```python
def translate(
    self,
    text: str,
    source_lang: str,
    target_lang: str,
    providers: Optional[List[str]] = None,
    models: Optional[Dict[str, str]] = None
) -> List[Translation]:
    """Translate text using multiple providers in parallel."""
    if source_lang not in self.supported_languages.values():
        raise ValueError(f"Source language '{source_lang}' not supported")
    if target_lang not in self.supported_languages.values():
        raise ValueError(f"Target language '{target_lang}' not supported")
    
    providers = providers or list(self.providers.keys())
    models = models or {}
    
    translations = []
    with ThreadPoolExecutor() as executor:
        future_to_provider = {
            executor.submit(
                self._get_provider_translation,
                provider,
                text,
                source_lang,
                target_lang,
                models.get(provider)
            ): provider
            for provider in providers
        }
        
        for future in as_completed(future_to_provider):
            try:
                translation = future.result()
                translations.append(translation)
            except Exception as e:
                provider = future_to_provider[future]
                print(f"Error getting translation from {provider}: {str(e)}")
    
    translations.sort(key=lambda x: x.confidence, reverse=True)
    return translations
```

This translate method is where everything comes together. It validates the languages, sets up parallel processing for all providers, and collects the results as they come in. We sort the translations by confidence score so the best ones appear first.

Finally, let's add some nice user interface touches:

```python
def print_slowly(text: str, delay: float = 0.03):
    """Print text with a slight delay between characters."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def display_translations(translations: List[Translation]):
    """Display translations in a formatted way."""
    print("\nTranslations:")
    print("-" * 50)
    for t in translations:
        print(f"\nProvider: {t.provider} ({t.model})")
        print(f"Translation: {t.text}")
        print(f"Confidence: {t.confidence:.2f}")
        print(f"Time taken: {t.time_taken:.2f}s")
        print("-" * 50)
```

These functions make our output more readable and add a nice typewriter effect for progress messages.

## Using the Translator

Save all this code in a file called `translator.py`. Here's how to use it:

```python
def main():
    print_slowly("Initializing Multi-Provider Translator...")
    translator = MultiProviderTranslator()
    
    print("\nAvailable languages:")
    for name, code in translator.supported_languages.items():
        print(f"  {name}: {code}")
    
    while True:
        try:
            print("\n" + "=" * 50)
            text = input("\nEnter text to translate (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
                
            source_lang = input("Enter source language code: ").lower()
            target_lang = input("Enter target language code: ").lower()
            
            print_slowly("\nGetting translations from all providers...")
            
            translations = translator.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            display_translations(translations)
            
        except ValueError as e:
            print(f"\nError: {str(e)}")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
```

Run it with:

```bash
python translator.py
```

## Making It Better

This translator can be enhanced in many ways. You might want to add language detection, so users don't have to specify the source language. You could add translation memory to cache common translations, or create a web interface instead of using the command line.

You could also make it smarter about choosing providers - maybe use cheaper ones for simple translations and reserve the more powerful models for complex text. Or add specialized vocabulary handling for technical translations.