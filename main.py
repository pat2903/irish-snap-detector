from irish_snap_detector import CardDetector

def main():
    irish_snap_game = CardDetector()

    try:
        irish_snap_game.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()