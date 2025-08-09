import chess.pgn
import io
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def convert_bb(x):
    bb_res = np.zeros((8, 8))
    for sq in x:
        r = 7 - sq // 8
        c = sq % 8
        bb_res[r][c] = 1
    return bb_res


def main():

    X, y = [], []

    for game in tqdm(open("games.pgn", "r").read().split("\n\n")[:10000]):
        game = chess.pgn.read_game(io.StringIO(game))
        winner = game.headers["Result"]
        board = game.board()
        if winner == "0-1" or winner == "1-0":
            winner = 0 if winner == "0-1" else 1 if winner == "1-0" else None
            for move in game.mainline_moves():
                board.push(move)
                bbs = []
                for colour in (chess.WHITE, chess.BLACK):
                    for piece in (chess.PAWN, chess.ROOK, chess.QUEEN, chess.KING, chess.KNIGHT, chess.BISHOP):
                        bb = list(board.pieces(piece, colour))
                        bbs.append(convert_bb(bb))
                bbs = np.array(bbs, dtype=np.uint64)
                bbs = np.ndarray.flatten(bbs)
                X.append(bbs)
                y.append(winner)

    model = NeuralNetwork()

    criterion = nn.BCELoss()

    LEARNING_RATE = 0.001
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    print("Finished creating model, beginning training")

    EPOCHS = 20
    for epoch in range(EPOCHS):
        print(f"Beginning epoch {epoch + 1}")
        model.train()
        total_loss = 0

        for batch in train_loader:
            X_batch, y_batch = batch

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                X_batch, y_batch = batch
                predictions = model(X_batch)
                test_loss += criterion(predictions, y_batch).item()

                predicted_labels = (predictions > 0.5).int()
                correct += (predicted_labels == y_batch).sum().item()
                total += y_batch.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total

        print(f"Training Loss: {avg_loss}\nTest Loss: {avg_test_loss}")
        print(f"Accuracy on test dataset: {round(accuracy * 100)}%")

    torch.save(model.state_dict(), "/Users/akshith/PycharmProjects/pythonProject49/model.pth")


if __name__ == "__main__":
    main()
