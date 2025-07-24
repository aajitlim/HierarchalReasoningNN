
---

# HierarchicalReasoningNN

This repository contains the implementation of a novel neural network architecture designed to solve complex reasoning tasks. Its core philosophy is a departure from standard, monolithic models, instead embracing a structured, two-pass system that mimics a process of **guess, reflect, and refine**.

The architecture is a proof-of-concept demonstrating how to tackle problems with a massive, combinatorial output space—scenarios where traditional methods fail due to the sheer scale of possibilities.

---

## The Central Philosophy: An Assembly of Specialists, Not a Single Genius

Imagine you need an answer to a highly complex question. You have two options:

1.  **The Monolithic Genius:** Ask a single person who claims to be an expert in everything. They will likely be overwhelmed, mix up details, and provide a mediocre, generalized answer.
2.  **The Specialist Team:** Give the problem to a junior analyst who performs a preliminary analysis. A manager then reviews this analysis, provides high-level feedback, and asks the analyst to redo the work with this new focus. The final result is far more accurate and refined.

This architecture is built on the second model. It posits that complex reasoning is not a single-shot act of brilliance, but an iterative process of focused inquiry guided by high-level oversight.

### The Two-Pass System: Guess, Reflect, Refine

The entire model operates on a two-pass workflow:

1.  **Pass 1: The Initial Analysis.** The `Worker` module, equipped with a structured, hierarchical memory, takes a first pass at the problem. It navigates its internal "filing cabinet" of knowledge to produce a preliminary understanding. This is the "first draft"—a reasonable but imperfect guess.

2.  **Pass 2: The Guided Refinement.** The `Supervisor` module acts as a manager. It does **not** look at the original, messy input. Instead, it examines the *output* of the first pass to get a high-level, "gestalt" understanding. From this vantage point, it generates a **guidance signal**—a piece of strategic advice. The `Worker` then performs its analysis a second time, but now its search is biased by the Supervisor's guidance, allowing it to focus its computational resources on the most promising avenues and ignore irrelevant distractions.

### The Visual Proof

This philosophy is not just theoretical. When tested on a synthetic task with **4,096 distinct classes**, the model demonstrates a remarkable ability to learn. The plot shows the model moving from near-zero accuracy to over 90% in just 10 epochs, confirming that this guided, iterative process is incredibly effective at navigating vast problem spaces.



---

## Architectural Breakdown

The system is composed of three conceptual components that work in concert.

### 1. The `HierarchicalMemoryWorker`: The Analyst with a Filing Cabinet

This is the core of the system. Its defining feature is its memory is not a single, flat lookup table but a structured hierarchy.

-   **Sectors (The Departments):** Memory is divided into broad `sectors`. The first step for the Worker is to determine which sector is most relevant to a given piece of input. This is the **Master Routing** decision.
-   **Memories (The Files):** Within each sector are specific `memories`. The Worker performs a second, local lookup to find the most relevant memories within the chosen sector.
-   **Dynamic Transformations:** Crucially, the memories are not static data points. They are learnable **transformation matrices**. The Worker doesn't just *retrieve* information; it *processes* its current understanding through these retrieved transformations, creating a new, richer representation.

### 2. The `SupervisorLayer`: The Meta-Cognitive Manager

The Supervisor provides the crucial reflective step.

-   **High-Level Abstraction:** It operates one level of abstraction above the Worker. It uses a Transformer to analyze the patterns and relationships in the Worker's initial output sequence.
-   **Strategic Guidance:** Its sole purpose is to produce a `guidance_signal`. This signal is a simple set of weights that advises the Worker on which memory sectors to prioritize during its second pass, effectively providing a "nudge" in the right direction.

### 3. The `HierarchicalClassifier`: The Orchestrator

This top-level module simply wires the components together into the two-pass sequence, ensuring the flow of information from Worker to Supervisor and back to the Worker for the final, refined output.

---

## Why This Architecture Matters

This project explores a design pattern with significant implications for future AI systems:

-   **Computational Efficiency:** By using sparse, guided activation (`top_k` lookups and supervisor guidance), the model avoids the costly brute-force approach of activating an entire massive network. It focuses its "attention" intelligently.
-   **Scalability:** The hierarchical and modular nature of the memory is inherently better suited to problems where the complexity or number of concepts can grow over time.
-   **Interpretability Potential:** The routing distributions (`sector_distribution`) offer a potential window into the model's "thought process," allowing us to see which categories of knowledge it considers relevant for a given input, both before and after supervisor guidance.

---

### A Deeper Dive into the Mechanics

To truly understand why this model succeeds, we must look beyond the high-level analogy and inspect the machinery itself. The elegance of the system lies in how simple, well-established operations like matrix multiplication and softmax are composed to create complex, emergent behavior.

#### Inside the `HierarchicalMemoryWorker`: A Journey of a Single Token

Let's trace the path of a single input token as it's processed by the Worker.

1.  **Embedding: From Word to Vector.**
    The process begins by converting the input token (an integer) into a dense vector using `nn.Embedding`. This transforms the token from a discrete symbol into a point in a continuous semantic space, where concepts can be related by distance and direction. This is our raw material, `x_emb`.

2.  **Master Routing: The Similarity Contest.**
    The core of the hierarchical system is deciding where to focus. This is achieved by comparing the input vector `x_emb` against a set of learnable `sector_keys`.
    -   `torch.matmul(x_emb, self.sector_keys.t())`: This is not just a mathematical operation; it's a parallel similarity contest. Each `sector_key` is a vector representing the "ideal" concept for that sector. The dot product measures the alignment or similarity between our input token and each of these sector concepts. A high score means high relevance.
    -   `F.softmax(sector_scores, dim=-1)`: The raw similarity scores are then passed through a softmax function. This normalizes them into a probability distribution, which can be interpreted as the model's **budget of attention**. If a sector gets a score of 0.7, it means the model will allocate 70% of its focus to the output generated by that sector.

3.  **Parallel Local Lookups: Checking Every File Cabinet Simultaneously.**
    Crucially, the model does **not** pick one sector and ignore the others. It investigates *every sector in parallel*. This allows it to gather evidence from multiple conceptual domains simultaneously. Inside each of the `num_sectors` loops:
    -   A **local routing** occurs, identical in principle to the master routing. The `x_emb` is compared to the `memory_keys` specific to that sector.
    -   This produces a `local_dist` over the memories within that sector.

4.  **Sparse Activation: The `top_k` Efficiency Hack.**
    Instead of dealing with all memories in a sector (which could be numerous), the model uses `torch.topk` to select only the `k` most relevant ones. This is a form of **sparse activation**. It forces the model to be decisive and focus its limited computational power on the most promising pieces of knowledge, dramatically improving efficiency.

5.  **Retrieving Transformations, Not Values.**
    This is perhaps the most critical concept. The `knowledge_matrix` does not store static values to be retrieved. It stores a collection of **learnable transformation matrices**.
    -   When the model retrieves the top `k` memories, it's not retrieving answers; it's retrieving **tools**. Think of it like a mechanic who, based on the problem, pulls a specific wrench, a screwdriver, and pliers from a toolbox.
    -   `torch.matmul(token_query.unsqueeze(...), retrieved_transformations)`: The input token (projected into a `query`) is then "operated on" by these retrieved tools. Each transformation matrix modifies the query vector in a unique way. The model learns for each memory what kind of transformation is most useful for solving the task.

6.  **Contextualized Aggregation.**
    The outputs of the top `k` transformations are combined via a weighted sum, using their `top_k` softmax scores as weights. This produces a single, contextualized vector for that sector. This vector represents the "opinion" of that specific memory sector.

7.  **Final Blending and the Residual Connection.**
    -   The "opinions" from all sectors are stacked together. The final context vector is a weighted sum of these opinions, using the master `sector_distribution` from Step 2 as the weights. The final vector is a sophisticated cocktail, with ingredients mixed from every sector, but the proportions are dictated by the initial master routing decision.
    -   `output = self.norm(x_emb + projected_output)`: The model does not replace the original token embedding. It **adds** the retrieved context to it. This is a **residual connection**. It forces the model to learn an *update* or *modification* (`projected_output`) rather than having to reconstruct the token's entire meaning from scratch. This stabilizes training and makes learning far more efficient.

---

### The `Supervisor`: Seeing the Forest for the Trees

The Supervisor's role is simple yet profound: to provide high-level, strategic direction.

-   **Input is Key:** It receives the sequence of processed vectors from the Worker's first pass (`worker_output_pass1`). It has access to the Worker's initial interpretation of the entire sentence.
-   **The Transformer's Power:** It uses a `nn.TransformerEncoderLayer` to perform self-attention over this sequence. This allows it to discover long-range dependencies and relationships that the Worker, processing token-by-token, might have missed. It builds a holistic understanding of the sentence's structure and intent.
-   **Condensing to a Command:** The output of the Transformer is averaged (`.mean(dim=1)`) to create a single vector that summarizes the entire sentence's meaning. This summary is then passed through a simple linear layer (`self.recommender`) to produce the `guidance_signal`. This signal is perfectly shaped to be added directly to the master routing scores in the Worker's second pass, effectively acting as the "voice of the manager."

---

### Why the Synthetic Task is the Perfect Testbed

The `generate_harder_task_data` function is not arbitrary; it's specifically designed to pressure-test the core competencies of this architecture.

-   **Combinatorial Complexity:** The correct label depends on the **combination** of three distinct tokens (`context`, `keyword`, `modifier`). A model cannot succeed by just identifying one important word. It is forced to learn relationships, making the Transformer in the Supervisor and the blending mechanism in the Worker essential.
-   **Positional Invariance:** These three tokens are placed at random positions within a sequence of "filler" tokens. This defeats simple models that rely on word position (like a Recurrent Neural Network might). The architecture must learn to identify the key tokens regardless of where they appear.
-   **Signal vs. Noise:** The filler tokens are noise. The model must learn to ignore them. The two-pass system excels here: the first pass might give some attention to the noise, but the Supervisor, seeing the whole picture, can guide the second pass to focus only on the true signal carriers.

In essence, the task creates a scenario where a naive, one-shot approach is destined to fail, thereby proving that the structured, hierarchical, and reflective nature of this model is not just an interesting idea, but a necessary solution for this class of problem.
