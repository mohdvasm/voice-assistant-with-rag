from assistant import Assistant
from speech_utils import SpeechUtils

files = [
    "/home/vasim/Downloads/PDFs/M3-R4 Programming and Problem Solving through C.pdf"
]

assistant = Assistant()
speaker = SpeechUtils()

assistant.get_retriever_from_files(files)

if __name__ == "__main__":
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print(f"Assistant initialized with {len(files)} files.") 

    print("Welcome to the Assistant! Type 'exit' to quit.")
    stop_count = 0 
    
    while True:
        print(f"\n\n{'-'*50}")
        # query = input("\nQuery: ")
        query = speaker.listen()
        print(f"\nUser: {query}")
        if query.lower() == 'exit':
            print("\nExiting the Assistant. Goodbye!\n")
            break
        elif query == "" or "you" in query.lower():
            print("\nNo input detected. Please try again.")
            stop_count += 1
            if stop_count >= 3:
                print("\nToo many empty inputs. Exiting the Assistant.")
                break
            continue

        result, content = assistant(query)
        print(f"\nAssistant: {content}")
        speaker.speak(content)
        # print(f"\n\n{'-'*50}")