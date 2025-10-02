# 🪄 Samantha Assistant

Samantha is a natural language powered assistant that helps you interact with your system more easily.  
It can be used in your terminal on **openEuler** (and other Linux systems), which will translate natural language into executable commands, and execute them.  

---

## 🚀 Getting Started

### Step 1. Clone this repository
```bash
git clone https://github.com/compsoc-oe-week/track1_nottingDuck.git
```
### Step 2. Enter the `work` directory
```bash
cd track1_nottingDuck/work
```

### Step 3. Run the setup script
```bash
chmod +x setup.sh
./setup.sh
```
#### The setup script will:

- ✅ Ensure **python3** and **pip3** are installed on your system  

- ✅ Install all required Python dependencies listed in `requirements.txt`  

- ✅ Configure the shell so you can use `samantha` as a global command 

## 💻 Usage

Once installed, simply run:
```bash
samantha <your natural language command>
```

For example:
```bash
samantha create a file named test.txt in the current directory and write hello world to it
```