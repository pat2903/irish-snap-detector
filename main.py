from irish_snap_detector import CardDetector

def main():
    # note: change the value '1' to the desired external camera
    irish_snap_game = CardDetector(1)

    try:
        irish_snap_game.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()