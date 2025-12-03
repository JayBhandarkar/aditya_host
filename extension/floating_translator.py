"""
Floating Translator - Cultural Design
Circular button that expands to cultural translator interface
"""
import tkinter as tk
from tkinter import ttk
from tkinter import font
from PIL import Image, ImageTk
import os

# Handle optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

class FloatingTranslator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)
        
        # Cultural color scheme matching main site
        self.colors = {
            'primary': '#8B4513',        # Brown
            'secondary': '#CD853F',      # Sandy brown
            'accent': '#DAA520',         # Goldenrod
            'bg_primary': '#FFF8DC',     # Cornsilk
            'bg_secondary': '#F5DEB3',   # Wheat
            'text_primary': '#2F1B14',   # Dark brown
            'text_secondary': '#8B4513', # Brown
            'danger': '#CD5C5C'
        }
        
        self.api_url = "http://localhost:8000/translate"
        self.expanded = False
        self.create_floating_button()
        
        # Start as small circular button
        self.root.geometry('80x80+50+50')
    
    def create_floating_button(self):
        # Circular button with brown border using Canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=80, 
            height=80, 
            highlightthickness=0
        )
        self.canvas.configure(bg=self.root.cget('bg'))
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Brown outer circle
        self.outer_circle = self.canvas.create_oval(
            5, 5, 75, 75, 
            fill='#8B4513', 
            outline='#8B4513',
            width=2
        )
        
        # Load and display image
        try:
            if os.path.exists("unnamed.jpg"):
                img = Image.open("unnamed.jpg")
                img = img.resize((50, 50), Image.Resampling.LANCZOS)
                
                # Make circular mask
                mask = Image.new('L', (50, 50), 0)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, 50, 50), fill=255)
                img.putalpha(mask)
                
                self.photo = ImageTk.PhotoImage(img)
                self.image_item = self.canvas.create_image(
                    40, 40,
                    image=self.photo
                )
            else:
                self.text_item = self.canvas.create_text(
                    40, 40,
                    text="ST",
                    font=('Inter', 16, 'bold'),
                    fill='white'
                )
        except:
            self.text_item = self.canvas.create_text(
                40, 40,
                text="ST",
                font=('Inter', 16, 'bold'),
                fill='white'
            )
        
        self.canvas.configure(cursor='hand2')
        
        # Bind events to canvas
        self.canvas.bind('<Enter>', self.on_hover_enter)
        self.canvas.bind('<Leave>', self.on_hover_leave)
        self.canvas.bind('<ButtonRelease-1>', self.on_click)
        self.canvas.bind('<ButtonPress-1>', self.start_drag)
        self.canvas.bind('<B1-Motion>', self.drag)
    
    def on_hover_enter(self, event):
        self.canvas.itemconfig(self.outer_circle, fill='#CD853F')
    
    def on_hover_leave(self, event):
        self.canvas.itemconfig(self.outer_circle, fill='#8B4513')
    
    def on_click(self, event):
        if not hasattr(self, 'was_dragged') or not self.was_dragged:
            self.expand()
    
    def create_interface(self):
        # Cultural main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'], relief='solid', bd=2)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Cultural ornamental border
        accent_bar = tk.Frame(main_frame, bg=self.colors['accent'], height=8)
        accent_bar.pack(fill=tk.X)
        
        # Cultural header
        header = tk.Frame(main_frame, bg=self.colors['bg_secondary'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Cultural title with icon
        title_label = tk.Label(
            header,
            text="🏛️ Sanskriti Translator",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Inter', 16, 'bold')
        )
        title_label.pack(side=tk.LEFT, padx=25, pady=20)
        
        # Cultural control buttons
        controls_frame = tk.Frame(header, bg=self.colors['bg_secondary'])
        controls_frame.pack(side=tk.RIGHT, padx=20, pady=20)
        
        # Cultural minimize button
        minimize_btn = tk.Label(
            controls_frame,
            text="−",
            bg=self.colors['accent'],
            fg='white',
            font=('Inter', 14, 'bold'),
            cursor='hand2',
            padx=8,
            pady=4
        )
        minimize_btn.pack(side=tk.RIGHT, padx=(5, 0))
        minimize_btn.bind('<Button-1>', self.minimize)
        minimize_btn.bind('<Enter>', lambda e: minimize_btn.config(bg=self.colors['secondary']))
        minimize_btn.bind('<Leave>', lambda e: minimize_btn.config(bg=self.colors['accent']))
        
        # Cultural close button
        close_btn = tk.Label(
            controls_frame,
            text="×",
            bg=self.colors['accent'],
            fg='white',
            font=('Inter', 14, 'bold'),
            cursor='hand2',
            padx=8,
            pady=4
        )
        close_btn.pack(side=tk.RIGHT)
        close_btn.bind('<Button-1>', self.close_app)
        close_btn.bind('<Enter>', lambda e: close_btn.config(bg=self.colors['danger'], fg='white'))
        close_btn.bind('<Leave>', lambda e: close_btn.config(bg=self.colors['accent'], fg='white'))
        
        # Make header draggable
        header.bind('<ButtonPress-1>', self.start_drag)
        header.bind('<B1-Motion>', self.drag)
        title_label.bind('<ButtonPress-1>', self.start_drag)
        title_label.bind('<B1-Motion>', self.drag)
        
        # Cultural separator
        separator = tk.Frame(main_frame, bg=self.colors['primary'], height=3)
        separator.pack(fill=tk.X)
        
        # Cultural content area
        content = tk.Frame(main_frame, bg=self.colors['bg_primary'], padx=30, pady=25)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Cultural language selection
        lang_frame = tk.Frame(content, bg=self.colors['bg_secondary'], relief='solid', bd=2)
        lang_frame.pack(fill=tk.X, pady=(0, 25), ipady=8)
        
        tk.Label(
            lang_frame, 
            text="🌍 Language Selection", 
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_primary'],
            font=('Inter', 12, 'bold')
        ).pack(pady=(10, 10))
        
        lang_controls = tk.Frame(lang_frame, bg=self.colors['bg_secondary'])
        lang_controls.pack(pady=(0, 10))
        
        tk.Label(lang_controls, text="From:", bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'], font=('Inter', 10)).pack(side=tk.LEFT, padx=(0, 8))
        
        # Cultural combobox styling
        style = ttk.Style()
        style.configure('Cultural.TCombobox', fieldbackground=self.colors['bg_primary'])
        
        self.from_lang = ttk.Combobox(
            lang_controls, 
            values=["Nepali", "Sinhala"], 
            state="readonly", 
            width=12,
            style='Cultural.TCombobox',
            font=('Inter', 10)
        )
        self.from_lang.set("Nepali")
        self.from_lang.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(lang_controls, text="To:", bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'], font=('Inter', 10)).pack(side=tk.LEFT, padx=(0, 8))
        
        tk.Label(
            lang_controls,
            text="English",
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            font=('Inter', 10, 'bold'),
            relief='solid',
            bd=1,
            padx=10,
            pady=2
        ).pack(side=tk.LEFT)
        
        # Cultural input section
        input_section = tk.Frame(content, bg=self.colors['bg_secondary'], relief='solid', bd=2)
        input_section.pack(fill=tk.BOTH, expand=True, pady=(0, 20), ipady=15)
        
        tk.Label(
            input_section, 
            text="📝 Source Text", 
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_primary'],
            font=('Inter', 12, 'bold')
        ).pack(padx=20, pady=(10, 8), anchor=tk.W)
        
        self.source_text = tk.Text(
            input_section, 
            height=3, 
            font=('Inter', 11), 
            wrap=tk.WORD,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            relief='solid',
            bd=2,
            selectbackground=self.colors['accent'],
            selectforeground='white',
            insertbackground=self.colors['text_primary']
        )
        self.source_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Small translate button between source and output
        translate_btn_frame = tk.Frame(input_section, bg=self.colors['bg_secondary'])
        translate_btn_frame.pack(pady=5)
        
        self.translate_btn = tk.Label(
            translate_btn_frame,
            text="▼ TRANSLATE ▼",
            bg=self.colors['primary'],
            fg='white',
            font=('Inter', 11, 'bold'),
            cursor='hand2',
            padx=20,
            pady=8,
            relief='raised',
            bd=2
        )
        self.translate_btn.pack()
        self.translate_btn.bind('<Button-1>', self.translate)
        self.translate_btn.bind('<Enter>', lambda e: self.translate_btn.config(bg=self.colors['secondary']))
        self.translate_btn.bind('<Leave>', lambda e: self.translate_btn.config(bg=self.colors['primary']))
        
        # Cultural output section - RIGHT BELOW SOURCE
        output_section = tk.Frame(input_section, bg=self.colors['bg_secondary'])
        output_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 15))
        
        tk.Label(
            output_section, 
            text="✨ Translated Text", 
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_primary'],
            font=('Inter', 12, 'bold')
        ).pack(pady=(0, 8), anchor=tk.W)
        
        self.result_text = tk.Text(
            output_section, 
            height=8, 
            font=('Arial', 12, 'bold'), 
            wrap=tk.WORD,
            bg='#FFE4B5',
            fg='#2F1B14',
            relief='solid',
            bd=3
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert('1.0', 'Translation will appear here...')
        
        # Cultural copy button
        button_frame = tk.Frame(content, bg=self.colors['bg_primary'])
        button_frame.pack(fill=tk.X, pady=10)
        
        copy_btn = tk.Label(
            button_frame,
            text="📋 Copy Translation",
            bg=self.colors['secondary'],
            fg='white',
            font=('Inter', 10, 'bold'),
            cursor='hand2',
            padx=12,
            pady=6,
            relief='raised',
            bd=2
        )
        copy_btn.pack(pady=(10, 0))
        copy_btn.bind('<Button-1>', self.copy_result)
        copy_btn.bind('<Enter>', lambda e: copy_btn.config(bg=self.colors['accent']))
        copy_btn.bind('<Leave>', lambda e: copy_btn.config(bg=self.colors['secondary']))
        
        # Cultural status
        status_frame = tk.Frame(content, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.status_label = tk.Label(
            status_frame, 
            text="🏛️ Ready for Sacred Translation", 
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_secondary'],
            font=('Inter', 9)
        )
        self.status_label.pack(pady=8)
        
        # Keyboard shortcuts
        self.root.bind('<Control-Return>', self.translate)
        self.root.bind('<Control-c>', self.copy_result)
        self.root.bind('<Escape>', self.minimize)
        
        # Focus on input
        self.source_text.focus_set()
        
    def translate(self, event=None):
        text = self.source_text.get("1.0", tk.END).strip()
        if not text:
            self.status_label.config(text="📝 Please enter sacred text to transform")
            return
            
        self.status_label.config(text="✨ Performing sacred transformation...")
        self.translate_btn.config(text="Translating...", bg='#999')
        self.root.update()
        
        # Map display names to API codes
        lang_map = {
            "Nepali": "ne_NP", 
            "Sinhala": "si_LK"
        }
        
        src_lang = lang_map.get(self.from_lang.get(), "en_XX")
        tgt_lang = "en_XX"
        
        if not HAS_REQUESTS:
            # Fallback mock translations
            mock_translations = {
                "hello": "नमस्ते",
                "good morning": "शुभ प्रभात", 
                "thank you": "धन्यवाद"
            }
            result = mock_translations.get(text.lower(), f"[Demo Translation] {text}")
            
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", result)
            
            self.status_label.config(text="Demo mode - Install 'requests' for real translation")
            self.translate_btn.config(text="TRANSLATE", bg=self.colors['primary'])
            return
        
        # Real API call to backend
        try:
            payload = {
                "text": text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            }
            
            # Debug output without unicode issues
            print("Sending translation request...")
            print(f"Text length: {len(text)} characters")
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            
            print(f"Response status: {response.status_code}")
            if response.status_code == 200:
                print("Translation successful")
            else:
                print("Translation failed")
            
            if response.status_code == 200:
                response_data = response.json()
                result = response_data.get('translated_text', 'Translation failed')
                
                result = response_data.get('translated_text', 'Translation failed')
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert("1.0", result)
                self.result_text.config(bg='#E8F5E8')
                self.root.after(1500, lambda: self.result_text.config(bg='#FFE4B5'))
                self.status_label.config(text="✨ Sacred transformation complete!")
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.text else 'No response'
                self.status_label.config(text=f"⚠️ API Error: {response.status_code} - {error_detail}")
                
        except requests.exceptions.ConnectionError:
            self.status_label.config(text="⚠️ Cannot connect to translation service")
        except requests.exceptions.Timeout:
            self.status_label.config(text="⚠️ Translation timeout - try shorter text")
        except Exception as e:
            print("Translation exception occurred")
            self.status_label.config(text="Error: Translation failed")
        
        self.translate_btn.config(text="TRANSLATE", bg=self.colors['primary'])
    
    def close_app(self, event=None):
        self.root.quit()
        self.root.destroy()
    
    def copy_result(self, event=None):
        result = self.result_text.get("1.0", tk.END).strip()
        if result:
            if HAS_PYPERCLIP:
                pyperclip.copy(result)
            else:
                self.root.clipboard_clear()
                self.root.clipboard_append(result)
            self.status_label.config(text="📋 Sacred text copied to divine clipboard!")
        else:
            self.status_label.config(text="⚠️ No sacred text to copy")
        
    def start_drag(self, event):
        self.x = event.x
        self.y = event.y
        self.was_dragged = False
        
    def drag(self, event):
        # Calculate movement
        dx = abs(event.x - self.x)
        dy = abs(event.y - self.y)
        
        # If moved more than 3 pixels, it's a drag
        if dx > 3 or dy > 3:
            self.was_dragged = True
            x = self.root.winfo_x() + event.x - self.x
            y = self.root.winfo_y() + event.y - self.y
            self.root.geometry(f'+{x}+{y}')
    
    def expand(self, event=None):
        if not self.expanded:
            self.expanded = True
            
            # Get current button position
            current_x = self.root.winfo_x()
            current_y = self.root.winfo_y()
            
            # Position translator near button
            new_x = current_x + 90  # To the right
            new_y = current_y - 50  # Slightly above
            
            # Keep on screen
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            if new_x + 420 > screen_width:
                new_x = current_x - 420  # Left instead
            if new_y < 0:
                new_y = current_y + 90  # Below instead
            
            self.canvas.pack_forget()
            
            if not hasattr(self, 'main_interface'):
                self.create_interface()
                self.main_interface = True
            else:
                # Show existing interface
                for widget in self.root.winfo_children():
                    if widget != self.canvas:
                        widget.pack(fill=tk.BOTH, expand=True)
            
            # Cultural size
            self.root.geometry(f'450x550+{new_x}+{new_y}')
    
    def minimize(self, event=None):
        if self.expanded:
            self.expanded = False
            
            # Hide main interface
            for widget in self.root.winfo_children():
                if widget != self.canvas:
                    widget.pack_forget()
            
            # Show floating button at current position
            self.canvas.pack(fill=tk.BOTH, expand=True)
            current_x = self.root.winfo_x()
            current_y = self.root.winfo_y()
            self.root.geometry(f'80x80+{current_x}+{current_y}')
            self.was_dragged = False
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FloatingTranslator()
    app.run()