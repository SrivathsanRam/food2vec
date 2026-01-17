import { useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './RecipeGraph.css';

const RecipeGraph = ({ graph, recipeName }) => {
  // Convert graph data to React Flow format
  const { initialNodes, initialEdges } = useMemo(() => {
    if (!graph || !graph.nodes || !graph.edges) {
      return { initialNodes: [], initialEdges: [] };
    }

    const nodes = graph.nodes.map((node, index) => {
      // Calculate position based on node type
      let x, y;
      if (node.type === 'ingredient') {
        // Ingredients on the left, stacked vertically
        x = 50;
        y = index * 80;
      } else if (node.type === 'final') {
        // Final node on the right
        x = 450;
        y = 150;
      } else {
        // Intermediate nodes in the middle
        x = 250;
        y = (index - graph.metadata?.ingredient_count || 0) * 100 + 50;
      }

      return {
        id: String(node.id),
        position: { x, y },
        data: { 
          label: node.label,
          state: node.state,
          type: node.type
        },
        type: 'default',
        style: getNodeStyle(node.type, node.state),
      };
    });

    const edges = graph.edges.map((edge, index) => ({
      id: `e${edge.source}-${edge.target}-${index}`,
      source: String(edge.source),
      target: String(edge.target),
      label: edge.action,
      animated: true,
      style: { stroke: '#667eea' },
      labelStyle: { fill: '#333', fontWeight: 500, fontSize: 10 },
      labelBgStyle: { fill: 'white', fillOpacity: 0.8 },
    }));

    return { initialNodes: nodes, initialEdges: edges };
  }, [graph]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return (
      <div className="recipe-graph-empty">
        <p>No graph data available</p>
      </div>
    );
  }

  return (
    <div className="recipe-graph-container">
      <h4 className="graph-title">🔗 Recipe Flow: {recipeName}</h4>
      <div className="graph-wrapper">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          attributionPosition="bottom-left"
          minZoom={0.5}
          maxZoom={1.5}
        >
          <Background color="#aaa" gap={16} />
          <Controls />
          <MiniMap 
            nodeColor={(node) => {
              if (node.data?.type === 'ingredient') return '#10b981';
              if (node.data?.type === 'final') return '#f59e0b';
              return '#667eea';
            }}
          />
        </ReactFlow>
      </div>
      <div className="graph-legend">
        <span className="legend-item">
          <span className="legend-dot ingredient"></span> Ingredient
        </span>
        <span className="legend-item">
          <span className="legend-dot intermediate"></span> Step
        </span>
        <span className="legend-item">
          <span className="legend-dot final"></span> Final
        </span>
      </div>
    </div>
  );
};

function getNodeStyle(type, state) {
  const baseStyle = {
    padding: '10px 15px',
    borderRadius: '8px',
    fontSize: '12px',
    fontWeight: 500,
    border: '2px solid',
    minWidth: '100px',
    textAlign: 'center',
  };

  switch (type) {
    case 'ingredient':
      return {
        ...baseStyle,
        background: '#d1fae5',
        borderColor: '#10b981',
        color: '#065f46',
      };
    case 'final':
      return {
        ...baseStyle,
        background: '#fef3c7',
        borderColor: '#f59e0b',
        color: '#92400e',
      };
    default:
      return {
        ...baseStyle,
        background: '#e0e7ff',
        borderColor: '#667eea',
        color: '#3730a3',
      };
  }
}

export default RecipeGraph;
