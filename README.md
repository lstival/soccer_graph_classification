# soccer_graph_classification
Using machine learning pipeline to predict entry into the attack zone in football

Example Data: https://doi.org/10.6084/m9.figshare.19222746

With the player's position, we create a graph representation, where the vertices represent the players and the edges the possibility to players makes passes between them:

![til](./readme_images/graph_plus_real_match.gif)

This represetation allows us to exctract metrics from these complex networks, in this work 8 metrics were obtained:

- Betweenness Centrality
- Eccentricity
- Global Efficiency
- Local Efficiency
- Vulnerability
- Clustering Coefficient
- Entropy
- PageRank

So each frame of video was generated a graph with the metrics, the image below demonstrate an example where the size of each node changes conform the player's Betweenness Centrality:

![til](./readme_images/centralidade_grafo_jogadores.png)

How we analyze a interval of the initial 5 seconds, all the metrics extracted are converted in a visual rhythm image, where in x axis represents the pass of time and y the player's metrics value (high values with light tons and low values closes to black):

![til](./readme_images/sample_resized.png)

In this way is possible to represent the game time series in a unique image:

![til](./readme_images/graph_edges_plus_red.gif)

### Citing
If you use this for academic research, please cite it using the following BibTeX entry.
```bibtex
@article{stival2023using,
  title={Using machine learning pipeline to predict entry into the attack zone in football},
  author={Stival, Leandro and Pinto, Allan and Andrade, Felipe dos Santos Pinto de and Santiago, Paulo Roberto Pereira and Biermann, Henrik and Torres, Ricardo da Silva and Dias, Ulisses},
  journal={PloS one},
  volume={18},
  number={1},
  pages={e0265372},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
