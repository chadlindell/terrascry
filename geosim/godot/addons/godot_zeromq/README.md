# godot_zeromq_bin

binary repository of [godot_zeromq](https://github.com/funatsufumiya/godot_zeromq)

if you need `godot_zeromq` [releases](https://github.com/funatsufumiya/godot_zeromq/releases) as `git submodule`, please use this repository.

## Simple usage example

```bash
cd PROJECT_DIR_OF_YOUR_GODOT
cd addons
git submodule add https://github.com/funatsufumiya/godot_zeromq_bin.git zeromq
# or simply: git clone https://github.com/funatsufumiya/godot_zeromq_bin.git zeromq
```

but I recommend to use `godotenv addons install`. see [GodotEnv's readme](https://github.com/chickensoft-games/GodotEnv?tab=readme-ov-file#initializing-godotenv-in-a-project). GodotEnv's `addons.jsonc` config is below (partial):

```json
    "zeromq": {
      "url": "https://github.com/funatsufumiya/godot_zeromq_bin",
      "checkout": "v0.1.0",
    },
```
