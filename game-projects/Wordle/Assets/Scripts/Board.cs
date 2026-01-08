using System.Linq;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class Board : MonoBehaviour
{
    private static readonly KeyCode[] SUPPORTED_KEYS = new KeyCode[]
    {
        KeyCode.A, KeyCode.B, KeyCode.C, KeyCode.D, KeyCode.E, KeyCode.F,
        KeyCode.G, KeyCode.H, KeyCode.I, KeyCode.J, KeyCode.K, KeyCode.L,
        KeyCode.M, KeyCode.N, KeyCode.O, KeyCode.P, KeyCode.R, KeyCode.Q,
        KeyCode.S, KeyCode.T, KeyCode.U, KeyCode.V, KeyCode.W, KeyCode.X,
        KeyCode.Y, KeyCode.Z
    };

    private Row[] rows;

    private string[] solutions;
    private string[] validWords;
    private string word;

    private int rowIndex;
    private int rowNumber => rows.Length;

    private int tileIndex;
    private int tileNumber => rows[rowIndex].tiles.Length;

    [Header("States")]
    public Tile.State emptyState;
    public Tile.State occupiedState;
    public Tile.State correctState;
    public Tile.State wrongSpotState;
    public Tile.State incorrectState;

    [Header("UI")]
    public TextMeshProUGUI invalidWordText;
    public Button newWordButton;
    public Button tryAgainButton;

    private void Awake()
    {
        rows = GetComponentsInChildren<Row>();
    }

    private void Start()
    {
        LoadData();
        NewGame();
    }

    public void NewGame()
    {
        SetRandomWord();
        enabled = true;

        ClearBoard();
    }

    public void TryAgain()
    {
        enabled = true;

        ClearBoard();
    }

    private void LoadData()
    {
        TextAsset textFile = Resources.Load("official_wordle_all") as TextAsset;
        validWords = textFile.text.Split("\n");

        textFile = Resources.Load("official_wordle_common") as TextAsset;
        solutions = textFile.text.Split("\n");
    }

    private void SetRandomWord()
    {
        word = solutions[Random.Range(0, solutions.Length)];
        word = word.ToLower().Trim();
    }

    private void Update()
    {
        Row row = rows[rowIndex];

        if (Input.GetKeyDown(KeyCode.Backspace))
        {
            tileIndex = Mathf.Max(0, tileIndex - 1);
            row.tiles[tileIndex].SetLetter('\0');
            row.tiles[tileIndex].SetState(emptyState);

            invalidWordText.gameObject.SetActive(false);
        }
        else if(tileIndex >= tileNumber)
        {
            if (Input.GetKeyDown(KeyCode.Return))
            {
                SubmitRow(row);
            }
        }
        else
        {
            for(int i = 0; i < SUPPORTED_KEYS.Length; i++)
            {
                if (Input.GetKeyDown(SUPPORTED_KEYS[i]))
                {
                    row.tiles[tileIndex].SetLetter((char)SUPPORTED_KEYS[i]);
                    row.tiles[tileIndex].SetState(occupiedState);
                    tileIndex++;
                    break;
                }
            }
        }

    }

    private void SubmitRow(Row row)
    {
        if (!IsValidWord(row.word))
        {
            invalidWordText.gameObject.SetActive(true);
            return;
        }

        string remaining = word;

        for(int i = 0; i < tileNumber; i++)
        {
            Tile tile = row.tiles[i];

            if(tile.letter == word[i])
            {
                tile.SetState(correctState);

                // Remove correct letter and inset empty to kept it same lenght
                remaining = remaining.Remove(i, 1);
                remaining = remaining.Insert(i, " ");
            }
            else if (!word.Contains(tile.letter))
            {
                tile.SetState(incorrectState);
            }
        }

        for(int i = 0; i < tileNumber; i++)
        {
            Tile tile = row.tiles[i];

            if(tile.state != correctState && tile.state != incorrectState)
            {
                if (remaining.Contains(tile.letter))
                {
                    tile.SetState(wrongSpotState);

                    int index = remaining.IndexOf(tile.letter);
                    remaining = remaining.Remove(index, 1);
                    remaining = remaining.Insert(index, " ");
                }
                else
                {
                    tile.SetState(incorrectState);
                }
            }
        }

        if (HasWon(row))
        {
            enabled = false;

        }

        rowIndex++;
        tileIndex = 0;

        if(rowIndex >= rowNumber)
        {
            enabled = false;
        }
    }

    private bool IsValidWord(string word)
    {
        return validWords.ToList().Contains(word);
    }

    private bool HasWon(Row row)
    {
        for(int i = 0; i < tileNumber; i++)
        {
            if (row.tiles[i].state != correctState)
                return false;
        }

        return true;
    }

    private void ClearBoard()
    {
        foreach(Row row in rows)
        {
            foreach(Tile tile in row.tiles)
            {
                tile.SetLetter('\0');
                tile.SetState(emptyState);
            }
        }

        rowIndex = 0;
        tileIndex = 0;
    }

    private void OnEnable()
    {
        tryAgainButton.gameObject.SetActive(false);
        newWordButton.gameObject.SetActive(false);
    }

    private void OnDisable()
    {
        tryAgainButton.gameObject.SetActive(true);
        newWordButton.gameObject.SetActive(true);
    }
}
