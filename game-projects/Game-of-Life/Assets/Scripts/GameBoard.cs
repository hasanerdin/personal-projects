using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

public class GameBoard : MonoBehaviour
{
    [SerializeField] private Tilemap currentState;
    [SerializeField] private Tilemap nextState;
    [SerializeField] private Tile aliveTile;
    [SerializeField] private Tile deadTile;
    [SerializeField] private Pattern pattern;
    [SerializeField] private float updateInterval = 0.05f;

    private HashSet<Vector3Int> aliveCells;
    private HashSet<Vector3Int> cellsToCheck;

    public int population { get; private set; }
    public int iterations { get; private set; }
    public float time { get; private set; }

    private void Awake()
    {
        aliveCells = new HashSet<Vector3Int>();
        cellsToCheck = new HashSet<Vector3Int>();
    }

    private void Start()
    {
        SetPattern(pattern);
    }

    private void SetPattern(Pattern pattern)
    {
        Clear();

        Vector2Int center = pattern.GetCenter();

        for(int i=0; i<pattern.cells.Length; i++)
        {
            Vector3Int cell = (Vector3Int)(pattern.cells[i] - center);
            currentState.SetTile(cell, aliveTile);
            aliveCells.Add(cell);
        }

        population = aliveCells.Count;
    }

    private void Clear()
    {
        currentState.ClearAllTiles();
        nextState.ClearAllTiles();
        aliveCells.Clear();
        cellsToCheck.Clear();
        population = 0;
        iterations = 0;
        time = 0f;
    }

    private void OnEnable()
    {
        StartCoroutine(Simulate());
    }

    private IEnumerator Simulate()
    {
        // Every iteration new WaitForSeconds function creates new class wchich cause a garbage.
        // We can create it outside the loop but it starts with constant updateInterval.
        // It mean, when we change it while game is running, it wont be changed.

        var interval = new WaitForSeconds(updateInterval);
        yield return interval;

        while (enabled)
        {
            UpdateState();

            population = aliveCells.Count;
            iterations++;
            time += updateInterval;

            yield return interval;
        }
    }

    private void UpdateState()
    {
        cellsToCheck.Clear();

        // gather cells to check
        foreach(Vector3Int cell in aliveCells)
        {
            for(int x = -1; x <= 1; x++)
            {
                for(int y = -1; y <= 1; y++)
                {
                    Vector3Int offset = new Vector3Int(x, y, 0);
                    cellsToCheck.Add(cell + offset);
                }
            }
        }

        // transitioning cells to the next state
        foreach(Vector3Int cell in cellsToCheck)
        {
            int neighbors = CountNeighbors(cell);
            bool isAlive = IsAlive(cell);

            if(!isAlive && neighbors == 3)
            {
                nextState.SetTile(cell, aliveTile);
                aliveCells.Add(cell);
            }
            else if(isAlive && (neighbors < 2 || neighbors > 3))
            {
                nextState.SetTile(cell, deadTile);
                aliveCells.Remove(cell);
            }
            else
            {
                // Why?
                nextState.SetTile(cell, currentState.GetTile(cell));
            }
        }

        // Why?
        (nextState, currentState) = (currentState, nextState);
        nextState.ClearAllTiles();
    }

    private int CountNeighbors(Vector3Int cell)
    {
        int count = 0;

        for(int x = -1; x <= 1; x++)
        {
            for(int y = -1; y <= 1; y++)
            {
                Vector3Int offset = new Vector3Int(x, y, 0);
                Vector3Int neighborCell = cell + offset;

                // Current cell, it is not neighbor
                if(x == 0 && y == 0)
                {
                    continue;
                }
                else if (IsAlive(neighborCell))
                {
                    count++;
                }
            }
        }

        return count;
    }

    private bool IsAlive(Vector3Int cell)
    {
        return currentState.GetTile(cell) == aliveTile;
    }
}
