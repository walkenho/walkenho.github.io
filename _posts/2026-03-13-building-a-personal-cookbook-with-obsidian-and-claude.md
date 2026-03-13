---
title: >-
    Building a Personal Cookbook with Obsidian and Claude — Your Recipes, Your Way
header:
  overlay_image: /images/cookbooks.jpg
  overlay_filter: 0.5
  caption: "Photo by [Brett Jordan](https://unsplash.com/@brett_jordan) on [Unsplash](https://unsplash.com/photos/books-on-brown-wooden-shelf-34fc5V7pCoQ)"
  actions:
    - label: ""
classes: wide
tags:
  - obsidian
  - claude
  - AI
  - productivity
excerpt:
    How I used Obsidian's Dataview plugin and a Claude Code skill to turn a
    chaotic pile of recipes into a portable, personal cookbook
---

My recipe collection is a mess. It's a disorganized pile of physical cookbooks, links to cooking blogs, photos of recipes of other people's cookbooks, handwritten recipes from my mom and I don't know what else. And since I have lived in multiple countries, they are not even in a single language.

A few years ago, I made a first attempt at standardizing my recipes by hand-converting them into `.tex` files which I then compiled into a pdf-cookbook. Of course, this was not maintainable and I abandoned the idea quickly (leaving me with even more disconnected recipe data).  A second attempt consisted in taking photos of some of my favourite recipes, but that -again- just added to the pile of mess. 

With my attempts of digitization hampered, for a while I mostly relied on cookbooks plus some special digital recipes, but with lots of moving around happening recently, it was time to give this another attempt.

**If you want to build your own version of this, all the necessary information and files can be found in the [associated GitHub repository](https://github.com/walkenho/cookbook-creator-with-claude-and-obsidian)**. 

## Requirements

What I wanted was:

* A cookbook:
  * that lives on my own machine, but can be accessed from anywhere
  * that is written in a portable format
  * that allows me to sort my recipes in a way that is suitable for the moment, but stays flexible for the future
* and a workflow:
  * that standardizes recipe data from different sources into a common format
  * that preferably enables me to easily translate my recipes on the way

As it turns out, with the recent progress in LLMs, a workflow that uses [Claude Code](https://claude.ai/code) for data parsing and then stores the standardized recipes in [Obsidian](https://obsidian.md/) suits the bill just fine.

## Components: Obsidian and Claude

### Obsidian as a Recipe Store

[Obsidian](https://obsidian.md/) is a popular notetaking application in the second-brain and knowledge management community. Some of its advantages are:

* It uses markdown files. This means that you can easily move your notes somewhere else should you wish to do so. 
* Your notes live on your local device, but it has applications for all major operating systems. Since your notes are markdown, you can easily sync them across devices, giving you full control over your notes whilst having them accessible from wherever you go (even offline).
* Obsidian supports a vast ecosystem of plug-ins. 
* Obsidian supports [YAML frontmatter](https://frontmatter.codes/docs/markdown) and lets you query it using its [dataview query language (DQL)](https://blacksmithgu.github.io/obsidian-dataview/queries/structure/). YAML frontmatter are metadata blocks at the top of your files, where you can define tags and other metadata properties. The DQL then allows us to dynamically query our note metadata as if it was information in a database, creating dynamically self-updating tables and lists.

The last point is what I exploited for my cookbook. My cookbook consists of a simple folder that contains recipes which are tagged with `#recipe` plus some additional metadata I wanted to track for my recipes (whether they were vegetarian, prep and cooktime, which meal type they were). To pull all of this information together into a cookbook, I created a cookbook index page, that uses the DQL to pull together a dynamic view of all recipes and their properties sorted by meal type and recipe source.

### Claude Code + Recipe Skill for Recipe Parsing

[Claude Code](https://claude.ai/code) is a terminal-based AI assistant that lets you interact with the files on your local machine. One interesting aspect of Claude Code for this project is that it allows you to create your own [Claude code skills](https://docs.anthropic.com/en/docs/claude-code/skills) - reusable prompt templates that it will automatically evoke if needed. 

For this project, I created the Claude Code recipe skill, a skill that allows Claude to:

* Parse recipe images into text data.
* Translate recipes to English.
* Standardize the input recipe data and convert it into a structured markdown file according to a pre-defined template.
* Analyze the recipe data and subsequently applies the appropriate tags to the parsed recipe. 

You can find the [Claude recipe skill in my GitHub repository](https://github.com/walkenho/cookbook-creator-with-claude-and-obsidian)

Note that whilst here I used Claude because its skill feature makes it easy to build re-usable workflows, the same principle could be used with other LLMs.

## Prerequisites

* [Obsidian](https://obsidian.md/) installed; Dataview Community Plugin activated; a new folder to host your Obsidian vault (or use an existing one)
* [Claude Code](https://claude.ai/code) installed

## Building the Obsidian Cookbook Backbone

The Obsidian cookbook backbone consists of two components: (i) a collection of recipes with YAML frontmatter metadata and (ii) the cookbook index markdown file containing the dataview queries to pull the recipes together.

### Recipe File Structure

For the recipe files to be picked up by Obsidian's dataview query, each recipe must be tagged as `recipe`. I also used additional metadata fields to sort recipes into categories and add metadata of interest.

In order to later use Claude for parsing, it is helpful to define a standard template for your recipe that Claude can populate. I defined my recipe structure template as follows. Note the empty tickbox for each ingredient. Ticking these allows you to create a shopping list view of all required ingredients (but how exactly will be the topic of a future post).

```yaml
---
tags:
- recipe
- <category tag(s), e.g. breakfast, salad, soup, stew, vegetable, chicken, meat, fish, pasta, dessert>
- <vegetarian if vegetarian>
- new
serves: <number>
preptime: <number of minutes>
cooktime: <number of minutes>
source: <book or website name>
---

## Ingredients

- [ ] <ingredient 1>
- [ ] <ingredient 2>
…

## Comments

<Any notes, tips, or description about the dish. If none provided, omit this section entirely.>

## Steps

1. <Step 1>
2. <Step 2>

## Per Serving

<Only included if nutritional information is available in the source; omit this section otherwise.>

* <number> kcal
* <number>g protein
* <number>g fat
* <number>g carbohydrates
…
```

I used the following tags:

- **Category**: Sorts recipes into sections within the cookbook. 
- **Vegetarian**: Flags whether a recipe is vegetarian.
- **Other Tags**: 
    * `new`: Newly added, not yet tried recipe.
    * `quick`: Extra speedy recipe.
    * `favourite`: Personal favourite recipe that gets an extra flag in the cookbook.
  
In the Claude workflow, `category` and `vegetarian` will be defined by Claude, `new` will always be added and `quick` or `favourite` will be added manually based on my personal opinion.


### Defining the Cookbook Index

`Cookbook.md` - the cookbook index - is the backbone of the cookbook. It is a dynamic index that uses Obsidian's DQL to automatically index all of notes containing the `recipe` tag in your Obsidian vault. Its automatic querying dynamic means that when you drop a recipe into your Obsidian vault, the recipe will automatically get added to the appropriate section of your cookbook (given it has the right tags).

My cookbook index is organized into sections by either meal type (Breakfasts, Salads, Soups, etc.) or source (useful if you eg have a lot of recipes from a given cook book). Each section contains a table with the following columns:

- **Recipe Name**: links to the recipe file
- **🆕**: appears if the recipe has the `new` tag
- **⭐**: appears if the recipe has the `favourite` tag
- **🏃**: appears if the recipe has the `quick` tag
- **🥕**: appears if the recipe is `vegetarian`
- **🔪**: prep time in minutes
- **⏳**: total cook time (prep + cook)
- **📖**: source book or website

This is how a section of the cookbook index looks in practice: 

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/cookbook_md_example.png" alt="A Section of my Cookbook" style="max-width: 100%" class="center"/>
</figure>

### Example Queries

**Query By Meal Type**

This is how you create a table containing all breakfast recipes.

```markdown
TABLE WITHOUT ID file.link as "Recipe", (choice(contains(tags, "new"), "🆕", "")) as "", (choice(contains(tags, "favourite"), "⭐", "")) as "⭐", (choice(contains(tags, "quick"), "🏃", "")) as "🏃", (choice(contains(tags, "vegetarian"), "🥕", "")) as "🥕", preptime as "🔪", (default(preptime, 0) + default(cooktime, 0)) as "⏳", source as "📖"
FROM #recipe AND #breakfast
SORT file.name asc
```

**Query By Source**

In addition to sorting by meal type, I have sections for my favourite recipe sources. This is how you would query for all recipes from Hugh Fearnley-Whittingstall's *Much More Veg* book:

```markdown
TABLE WITHOUT ID file.link as "Recipe", (choice(contains(tags, "new"), "🆕", "")) as "", preptime as "🔪", (default(preptime, 0) + default(cooktime, 0)) as "⏳", source as "📖"
FROM #recipe 
WHERE source = "Much More Veg"
SORT file.name asc
```

**Customizing & Extending**

Dataview is flexible. Some ideas for further sections:

- Quick weeknight dinners (filter by `quick` + `favourite`)
- Seasonal recipes (add a `season` property and query by it)
- Recipes below xxx kcal or with more than xxx g protein.


## Building the Claude Recipe Skill 

Once I had the cookbook index backbone, I needed to populate it with recipes. For this, I created a Claude recipe skill.

This is how it looks like

````markdown
---
name: recipe
description: Format recipe data into a structured markdown file with YAML frontmatter, ingredients checklist, comments, and steps sections.
argument-hint: [recipe data, file path, or description]
---

Format the provided recipe data as a markdown file and save it.

## Input

The recipe data may arrive in any of these forms — handle all of them:

- **Pasted raw text** (e.g. copied from a website or book): parse it to extract the fields below.
- **Structured data** (JSON, YAML, or similar): read the fields directly.
- **File path**: read the file at that path first, then parse.
- **Interactive / incomplete input**: if critical fields are missing (title, ingredients, steps), ask the user to provide them before proceeding.

The recipe data is: $ARGUMENTS

## Output format

Save the recipe as a markdown file. Use the recipe title with appropriate title casing as filename, e.g. `Broccolini Soba Noodle Salad.md`. Save it in the current working directory unless the user specifies otherwise.

The file must follow this exact structure:

```
---
tags:
- recipe
- <category tag(s), e.g. breakfast, salad, soup, stew, vegetable, chicken, meat, fish, pasta, dessert>
- <vegetarian if vegetarian>
- new
serves: <number>
preptime: <number of minutes>
cooktime: <number of minutes>
source: <book or website name>
---

## Ingredients

- [ ] <ingredient 1>
- [ ] <ingredient 2>
…

## Comments

<Any notes, tips, or description about the dish. If none provided, omit this section entirely.>

## Steps

1. <Step 1>
2. <Step 2>

## Per Serving

* <number> kcal
* <number>g protein
* <number>g fat
* <number>g carbohydrates
…
```

## Rules

- ALWAYS stay true to the original data, never change the text, never make things up.
- Tags: always include `recipe`. Always include `new`. Include `vegetarian` if the dish is vegetarian. Infer category tags from the ingredients and dish type. Choose from: breakfast, salad, soup, stew, vegetable, chicken, meat, fish, pasta, dessert, dressing. Add dish tags like `high-protein`, `meal-prep` where appropriate. Only add tags that are clearly justified. 
- `serves`, `preptime`, `cooktime`: use integers.
- `preptime` is active preparation, cooktime is additional time cooking. If only one is provided, list it as `preptime`. If `preptime`/`cooktime` are not provided, omit the fields.
- `source`: use the book title, website name, or "Unknown" if not provided.
- Ingredients: use `- [ ]` checkboxes. Include quantities and units. 
- Steps: use a numbered list. Be clear and concise.
- Per Serving: Include kcal, protein and fat if present. If data is not present, omit this section.
- After saving the file, confirm the filename and path to the user.

````

Note: In order to use a skill, save the file as `~/.claude/skills/recipe/SKILL.md`. This registers it as a skill. You will be able to use it after restarting the conversation.

## Using Claude's Recipe Skill

To use Claude's new recipe skill:

1. Start a Claude Code conversation and give it your recipe source — paste in text from a webpage, attach a photo of a cookbook page, or type out your own notes. Then ask Claude to format it as a recipe, e.g. *"Convert this into a recipe file."*
2. Save the resulting file in your Obsidian vault. As long as it has the `recipe` tag, `Cookbook.md` will pick it up automatically.

**A word of caution**: Although Claude does a very decent job of parsing the recipes, it does not always get the amounts in the ingredients list right, so **double-check once you have parsed the recipe**. I have also noticed that when parsing multiple recipes at once, it will sometimes start putting ingredients from one recipe into another one, so better clear the context after each recipe.


## Final Thoughts

Overall I am quite happy with the performance of this workflow. Is it perfect? Certainly not. There are still manual steps involved and the double-checking still takes time. But it certainly is a very viable form of organizing a cookbook. I have used it to finally take advantage of my old hand-copied recipes. I have managed to make all my favourite recipes accessible on my mobile phone. And I finally have the feeling of having all my recipes accessible in a single place. 

If you'd like to try it yourself, everything you need is in my
[GitHub repository](https://github.com/walkenho/cookbook-creator-with-claude-and-obsidian).

Your recipes, your format, your device — but shareable anywhere. So you can have your cake and eat it, too. 

Happy cooking!
