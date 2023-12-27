data = {
    "common": {
        "style": "padding:5px;color:#ffffff;font-size: 18px;border-width: 1px;border-style: solid;"
                 "border-color: #ffffff;"
    },
    "mainwindow": {
        "TITLE": "Font Creator",
        "bg": "rgb(31,39,72)",
        "ICON": "static/pic/Pencil.svg"
    },
    "predict": {
        "style": "background-color:rgb(41,49,82);padding:5px;color:#ffffff;font-size: 32px;"
                 "border-width: 1px;border-style: solid;border-color: #ffffff;"
    },
    "menus": {
        "bg": "rgb(54,64,95)",
        "width": 200,
        "radius": "4px"
    },
    "main_logo": {
        "width": 160,
        "height": 80,
        "picpath": "static/pic/logo.png",
        "picpathm": "static/pic/logom.png"
    },
    "folded_logo": {
        "width": 48,
        "height": 48,
    },
    "menu_bg": {
        "width": 170,
        "height": 40,
        "enter": "rgb(81,93,128)",
        "leave": "rgba(81,93,128,0)",
        "press": "rgba(101,113,148,0)"
    },
    "folded_menu_icon": {
        "width": 50,
        "height": 40,
    },
    "menu_icon": {
        "painting": "static/pic/pencil.svg",
        "file": "static/pic/file.svg",
        "generate": "static/pic/generate.svg",
        "setting": "static/pic/setting.svg",
        "about": "static/pic/about.svg",
    },
    "menu_btn": {
        "color": "#ffffff",
        "font": "20px",
        "open": "static/pic/open.svg",
        "close": "static/pic/close.svg"
    },
    "button": {
        "border-width": "2px",
        "bg": "rgb(31,39,72)",
    },
    "topnav": {
        "bg": "rgb(54,64,95)",
        "radius": "4px",
        "width": 200,
        "mini": "static/pic/minus-circle.svg",
        "full": "static/pic/square.svg",
        "close": "static/pic/x-circle.svg",
    },
    "current": {
        "label": "画板",
        "color": "#ffffff"
    },
    "to_btn": {
        "width": 32,
        "height": 32,
        "focusbackground": "rgba(81,93,128,1)",
        "nobackground": "rgba(81,93,128,0)"
    },
    "description": {
        "text": """Version: 1.0.0
Author: 李鹏达 吴泽霖 张耘彪 武泽恺
鸣谢：良育老师
        """
    },
    "license": {
        "file": "static/license.txt",
    },
    "scroll": {
        "style": " QLabel {"
                 "   padding: 5px;"
                 "   color: #ffffff;"
                 "   font-size: 18px;"
                 "   border-width: 1px;"
                 "   border-style: solid;"
                 "   border-color: #ffffff;"
                 " }"
                 " QScrollArea {"
                 "   color: #ffffff;"
                 "   font-size: 18px;"
                 "   border-width: 1px;"
                 "   border-style: solid;"
                 "   border-color: #ffffff;"
                 " }"
                 " QTextEdit {"
                 "   padding: 5px;"
                 "   color: #ffffff;"
                 "   font-size: 18px;"
                 "   border-width: 1px;"
                 "   border-style: solid;"
                 "   border-color: #ffffff;"
                 " }"
                 " QScrollBar:vertical {"
                 "	border: none;"
                 "    background: rgb(52, 59, 72);"
                 "    width: 15px;"
                 "    margin: 21px 0 21px 0;"
                 "	border-radius: 0px;"
                 " }"
                 " QScrollBar::handle:vertical {	"
                 "	background: rgb(190 ,190 ,190);"
                 "    min-height: 25px;"
                 "	border-radius: 4px"
                 " }"
                 " QScrollBar::add-line:vertical {"
                 "     border: none;"
                 "    background: rgb(55, 63, 77);"
                 "     height: 20px;"
                 "	border-bottom-left-radius: 4px;"
                 "    border-bottom-right-radius: 4px;"
                 "     subcontrol-position: bottom;"
                 "     subcontrol-origin: margin;"
                 " }"
                 " QScrollBar::sub-line:vertical {"
                 "	border: none;"
                 "    background: rgb(55, 63, 77);"
                 "     height: 20px;"
                 "	border-top-left-radius: 4px;"
                 "    border-top-right-radius: 4px;"
                 "     subcontrol-position: top;"
                 "     subcontrol-origin: margin;"
                 " }"
                 " QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {"
                 "     background: none;"
                 " }"
                 " QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
                 "     background: none;"
                 " }"
                 " QScrollBar::handle:vertical {	"
                 "	background: rgb(190 ,190 ,190);"
                 "    min-height: 25px;"
                 "	border-radius: 4px"
                 " }"
                 " QScrollBar::handle:vertical:hover {"
                 "    background-color: rgb(211 ,211 ,211);"
                 " }"
                 " QScrollBar::handle:vertical:pressed {"
                 "    background-color: rgb(255 ,250 ,250);"
                 " }"
    },
    "right": [{"title": "画板"},
              {"title": "图库"},
              {"title": "生成"},
              {"title": "设置"},
              {"title": "关于"}],
    "menu": [{
        "name": "画板",
        "type": "painting"
    }, {
        "name": "图库",
        "type": "file"
    }, {
        "name": "生成",
        "type": "generate"
    }, {
        "name": "设置",
        "type": "setting"
    }, {
        "name": "关于",
        "type": "about"
    }],
    "show": {
        "style": "color:#ffffff;font-size: 18px;border-width: 1px;border-style: solid;border-color: #ffffff;",
    },
    "draw": {
        "path": "draw",
    },
    "slider": "QSlider::groove:horizontal {"
              "    border-radius: 5px;"
              "    height: 10px;"
              "	margin: 0px;"
              "	background-color: rgb(52, 59, 72);"
              "}"
              "QSlider::groove:horizontal:hover {"
              "	background-color: rgb(55, 62, 76);"
              "}"
              "QSlider::handle:horizontal {"
              "    background-color: rgb(190 ,190 ,190);"
              "    border: none;"
              "    height: 20px;"
              "    width: 20px;"
              "    margin: 0px;"
              "	border-radius: 5px;"
              "}"
              "QSlider::handle:horizontal:hover {"
              "    background-color: rgb(211 ,211 ,211);"
              "}"
              "QSlider::handle:horizontal:pressed {"
              "    background-color: rgb(255 ,250 ,250);"
              "}",
    "combo": "QComboBox{"
             "	background-color: rgb(27, 29, 35);"
             "	border-radius: 5px;"
             "	border: 2px solid rgb(33, 37, 43);"
             "	padding: 5px;"
             "	padding-left: 10px;"
             "  color: rgb(255, 255, 255);"
             "  font-size: 18px;"
             "}"
             "QComboBox:hover{"
             "	border: 2px solid rgb(64, 71, 88);"
             "}"
             "QComboBox::drop-down {"
             "	subcontrol-origin: padding;"
             "	subcontrol-position: top right;"
             "	width: 25px; "
             "	border-left-width: 3px;"
             "	border-left-color: rgba(39, 44, 54, 150);"
             "	border-left-style: solid;"
             "	border-top-right-radius: 3px;"
             "	border-bottom-right-radius: 3px;	"
             "	background-position: center;"
             "	background-repeat: no-reperat;"
             " }"
             "QComboBox QAbstractItemView {"
             "	color: rgb(170, 170, 170);	"
             "	background-color: rgb(33, 37, 43);"
             "	padding: 10px;"
             "	selection-background-color: rgb(39, 44, 54);"
             "}"
             "",
    "checkbox": "QCheckBox {"
                "    color: rgb(255, 255, 255);"
                "    font-size: 18px;"
                "}"
                "QCheckBox::indicator {"
                "    border: 3px solid rgb(52, 59, 72);"
                "    width: 15px;"  
                "    height: 15px;" 
                "    border-radius: 10px;"
                "    background: rgb(44, 49, 60);"
                "}" 
                "QCheckBox::indicator:hover {"  
                "    border: 3px solid rgb(58, 66, 81);"
                "}"
                "QCheckBox::indicator:checked {"
                "    background: 3px solid rgb(52, 59, 72);"
                "    border: 3px solid rgb(52, 59, 72);"
                "    background-image: url(static/pic/checked.png);"
                "}"
                "QCheckBox::indicator:checked:hover {"
                "    border: 3px solid rgb(58, 66, 81);"    
                "}"
                "QCheckBox::indicator:unchecked:hover {"
                "    border: 3px solid rgb(58, 66, 81);"
                "}"
                "QCheckBox::indicator:unchecked {"
                "    background: 3px solid rgb(52, 59, 72);"    
                "}"
}
