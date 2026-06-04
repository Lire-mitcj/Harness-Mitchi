CREATE OR REPLACE VIEW v_boarding_pass AS
SELECT passenger_id, flight_no FROM boarding;

CREATE PROCEDURE sp_check_in(p_id INT)
BEGIN
    SELECT 1;
END;

ALTER VIEW v_boarding_pass AS SELECT * FROM boarding;
