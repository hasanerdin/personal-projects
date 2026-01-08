🎮 2048 Clone in Unity – Project Overview

During the development of my 2048 clone using Unity, I focused not only on replicating the game's core mechanics, but also on applying and learning essential Unity concepts that are widely used in professional game development. This project helped me structure code more modularly, organize the UI efficiently, and understand deeper event-driven systems.

Here are the key Unity features and concepts I applied throughout the project:

🔧 Tools & Systems Used
🎨 Asset Importing & TextMeshPro

I imported and managed custom assets, including fonts and UI elements. TextMeshPro was used for sharp and scalable in-game text, such as tile values and score displays.

🧱 Canvas & UI Layouts

Built a responsive UI using Canvas, with Vertical and Horizontal Layout Groups to organize game tiles and interface elements dynamically based on resolution.

📦 Prefab System

Each tile in the game is a Prefab — allowing me to instantiate, update, and destroy tile objects efficiently and cleanly during gameplay.

📄 ScriptableObject

I used a ScriptableObject to define tile data, enabling a centralized and reusable way to manage tile configurations and properties like color or value.

🎮 Gameplay Architecture
📲 Input System

Input handling is done through both keyboard controls and mobile swipe detection. I also implemented an event-driven system using EventHandlers, making the input system modular and decoupled from the game logic.

🔁 Coroutines

Smooth animations and timed events (such as merging tiles or delaying Game Over screens) are handled using Unity’s Coroutine system.

🚦 Game States: Start & Game Over

The game includes a clean Start screen, and automatically triggers a Game Over screen when no moves are left. I managed game states via centralized logic to ensure clarity and scalability.

🕹️ UI Buttons & Events

Buttons like Restart, Home, and Continue were implemented using UnityEvents, and hooked into the core logic via listeners for responsive UI interaction.

🧠 What I Learned

How to design scalable UI with Unity’s layout system

* Prefab-based design for dynamic content
* Using ScriptableObjects to separate data from logic
* Event-driven programming with EventHandler for input separation
* Basic mobile support with swipe gestures
* Coroutine-based animations and transitions

Game state management and clean UI control

🔗 Play
🕹️ Play the game: [Your itch.io link here](https://hasanerdin.itch.io/2048)
