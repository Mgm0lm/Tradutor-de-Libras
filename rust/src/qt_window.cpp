#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstddef>
#include <cstdint>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtGui/QShortcut>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStyle>
#include <QtWidgets/QVBoxLayout>
#include <QtCore/QTimer>

struct WindowSettings {
    const char *title;
    const char *stylesheet;
    const std::uint32_t *camera_ids;
    const char *const *camera_names;
    std::size_t camera_count;
    int desync;
    int tts;
    int show_fps;
};

class Window final : public QMainWindow {
public:
    using Tick = void (*)(void *, Window *);
    using Action = void (*)(void *, int, int);

    QLabel *video;
    QLabel *fps;
    QLabel *letter;
    QLabel *status;
    QLabel *threshold;
    QProgressBar *progress;

    Window(const WindowSettings &settings, void *state, Tick tick, Action action) {
        setWindowTitle(QString::fromUtf8(settings.title));
        setStyleSheet(QString::fromUtf8(settings.stylesheet));

        auto *root = new QWidget;
        root->setObjectName("root");
        setCentralWidget(root);
        auto *outer = new QHBoxLayout(root);
        outer->setContentsMargins(24, 24, 24, 24);
        outer->setSpacing(24);

        auto *left = new QVBoxLayout;
        left->setSpacing(12);
        video = new QLabel;
        video->setObjectName("videoLabel");
        video->setAlignment(Qt::AlignCenter);
        video->setMinimumSize(320, 240);
        video->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        left->addWidget(video, 1);

        auto *controls = new QFrame;
        controls->setObjectName("card");
        auto *controls_layout = new QHBoxLayout(controls);
        controls_layout->setContentsMargins(14, 10, 14, 10);
        controls_layout->addWidget(new QLabel(QStringLiteral("Câmera:")));
        auto *camera = new QComboBox;
        for (std::size_t i = 0; i < settings.camera_count; ++i) {
            camera->addItem(QString::fromUtf8(settings.camera_names[i]), settings.camera_ids[i]);
        }
        connect(camera, &QComboBox::currentIndexChanged, this, [=](int index) {
            if (index >= 0) action(state, 0, camera->itemData(index).toInt());
        });
        controls_layout->addWidget(camera);
        controls_layout->addStretch(1);

        auto *desync = new QCheckBox(QStringLiteral("Dessincronizar câmera / IA"));
        desync->setChecked(settings.desync != 0);
        desync->setToolTip(QStringLiteral("Ligado: vídeo sempre fluido, a letra pode atrasar um pouco.\n"
                                         "Desligado: vídeo só atualiza junto com cada nova inferência."));
        connect(desync, &QCheckBox::toggled, this, [=](bool checked) { action(state, 1, checked); });
        controls_layout->addWidget(desync);
        controls_layout->addStretch(1);
        controls_layout->addWidget(new QLabel(QStringLiteral("Confiança mín.:")));

        auto *minus = new QPushButton("-");
        minus->setObjectName("thresholdBtn");
        connect(minus, &QPushButton::clicked, this, [=] { action(state, 2, -1); });
        controls_layout->addWidget(minus);
        threshold = new QLabel("0.70");
        controls_layout->addWidget(threshold);
        auto *plus = new QPushButton("+");
        plus->setObjectName("thresholdBtn");
        connect(plus, &QPushButton::clicked, this, [=] { action(state, 2, 1); });
        controls_layout->addWidget(plus);
        left->addWidget(controls);

        fps = new QLabel;
        fps->setObjectName("subtitulo");
        fps->setVisible(settings.show_fps != 0);
        left->addWidget(fps);
        outer->addLayout(left, 6);

        auto *right = new QVBoxLayout;
        right->setSpacing(16);
        auto *title = new QLabel(QStringLiteral("Letra reconhecida"));
        title->setObjectName("titulo");
        right->addWidget(title);

        auto *card = new QFrame;
        card->setObjectName("card");
        auto *card_layout = new QVBoxLayout(card);
        card_layout->setAlignment(Qt::AlignCenter);
        card_layout->setContentsMargins(20, 20, 20, 20);
        letter = new QLabel(QStringLiteral("–"));
        letter->setObjectName("letraLabel");
        letter->setAlignment(Qt::AlignCenter);
        card_layout->addWidget(letter);
        progress = new QProgressBar;
        progress->setRange(0, 100);
        progress->setValue(0);
        progress->setTextVisible(false);
        card_layout->addWidget(progress);
        status = new QLabel(QStringLiteral("Aguardando modelo..."));
        status->setObjectName("statusLabel");
        status->setAlignment(Qt::AlignCenter);
        card_layout->addWidget(status);
        right->addWidget(card, 1);

        auto *bottom = new QHBoxLayout;
        auto *tts = new QCheckBox(QStringLiteral("Falar em voz alta (TTS)"));
        tts->setChecked(settings.tts != 0);
        connect(tts, &QCheckBox::toggled, this, [=](bool checked) { action(state, 3, checked); });
        bottom->addWidget(tts);
        bottom->addStretch(1);
        auto *repeat = new QPushButton("Repetir");
        repeat->setObjectName("secondaryBtn");
        connect(repeat, &QPushButton::clicked, this, [=] { action(state, 4, 0); });
        bottom->addWidget(repeat);
        auto *quit = new QPushButton("Sair (ESC)");
        quit->setObjectName("secondaryBtn");
        connect(quit, &QPushButton::clicked, this, &QWidget::close);
        bottom->addWidget(quit);
        right->addLayout(bottom);
        outer->addLayout(right, 4);

        auto *escape = new QShortcut(QKeySequence(Qt::Key_Escape), this);
        connect(escape, &QShortcut::activated, this, &QWidget::close);
        auto *timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, [=] { tick(state, this); });
        timer->start(16);
    }
};

extern "C" int libras_ui_run(const WindowSettings *settings, void *state,
                            Window::Tick tick, Window::Action action) {
    SetProcessDPIAware();
    int argc = 1;
    char name[] = "TradutorLibras";
    char *argv[] = {name, nullptr};
    QApplication app(argc, argv);
    Window window(*settings, state, tick, action);
    window.showFullScreen();
    return app.exec();
}

extern "C" void libras_ui_state(Window *window, const char *letter, int progress,
                               const char *status, const char *fps, float threshold) {
    const QString text = letter ? QString::fromUtf8(letter) : QStringLiteral("–");
    const QString state = !letter ? "nada" : (progress >= 100 ? "confirmado" : "detectando");
    window->letter->setText(text);
    if (window->letter->property("estado").toString() != state) {
        window->letter->setProperty("estado", state);
        window->letter->style()->unpolish(window->letter);
        window->letter->style()->polish(window->letter);
    }
    window->progress->setValue(progress);
    window->status->setText(QString::fromUtf8(status));
    window->fps->setText(QString::fromUtf8(fps));
    window->threshold->setText(QString::number(threshold, 'f', 2));
}

extern "C" void libras_ui_frame(Window *window, const unsigned char *rgb, int width, int height,
                               const float *landmarks, int confirmed) {
    QImage image(rgb, width, height, width * 3, QImage::Format_RGB888);
    if (landmarks) {
        image = image.copy();
        QPainter painter(&image);
        const auto color = confirmed ? QColor(80, 220, 80) : QColor(40, 130, 235);
        constexpr int connections[][2] = {
            {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},{11,12},
            {9,13},{13,14},{14,15},{15,16},{13,17},{0,17},{17,18},{18,19},{19,20}
        };
        const auto point = [&](int i) {
            return QPoint(int(landmarks[i * 3] * width), int(landmarks[i * 3 + 1] * height));
        };
        painter.setPen(QPen(color, 2));
        for (const auto &edge : connections) painter.drawLine(point(edge[0]), point(edge[1]));
        painter.setPen(Qt::NoPen);
        painter.setBrush(color);
        for (int i = 0; i < 21; ++i) painter.drawEllipse(point(i), 4, 4);
    }
    window->video->setPixmap(QPixmap::fromImage(image).scaled(window->video->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
