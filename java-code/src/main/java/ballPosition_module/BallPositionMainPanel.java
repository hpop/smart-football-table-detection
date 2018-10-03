package ballPosition_module;

import java.awt.BorderLayout;
import java.io.IOException;

import javax.swing.JPanel;

public class BallPositionMainPanel extends JPanel {

	FieldPanel fieldPanel;
	SliderPanel sliderPanel;
	
	BallPositionController ballPositionController;

	public BallPositionMainPanel(BallPositionController ballPositionController) throws IOException {
		this.ballPositionController = ballPositionController;
		init();
	}

	private void init() throws IOException {

		fieldPanel = new FieldPanel(this);
		sliderPanel = new SliderPanel(this);

		this.setLayout(new BorderLayout());

		this.add(fieldPanel, BorderLayout.CENTER);
		this.add(sliderPanel, BorderLayout.SOUTH);

	}

	public FieldPanel getFieldPanel() {
		return fieldPanel;
	}

	public void setFieldPanel(FieldPanel fieldPanel) {
		this.fieldPanel = fieldPanel;
	}

	public SliderPanel getSliderPanel() {
		return sliderPanel;
	}

	public void setSliderPanel(SliderPanel sliderPanel) {
		this.sliderPanel = sliderPanel;
	}

	public BallPositionController getBallPositionController() {
		return ballPositionController;
	}

	public void setBallPositionController(BallPositionController ballPositionController) {
		this.ballPositionController = ballPositionController;
	}

}
